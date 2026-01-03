import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import e2cnn.nn as enn
from e2cnn import gspaces
from tqdm import tqdm
import os

# ============================================================
# Model definition
# ============================================================
class SO2InvariantCNN(nn.Module):
    def __init__(self, N=8):
        super().__init__()
        self.gspace = gspaces.Rot2dOnR2(N=N)

        in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        out1 = enn.FieldType(self.gspace, 8 * [self.gspace.regular_repr])
        out2 = enn.FieldType(self.gspace, 16 * [self.gspace.regular_repr])
        out3 = enn.FieldType(self.gspace, 16 * [self.gspace.regular_repr])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, out1, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out1),
            enn.ReLU(out1, inplace=True),
            enn.PointwiseMaxPool(out1, 2)
        )

        self.block2 = enn.SequentialModule(
            enn.R2Conv(out1, out2, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out2),
            enn.ReLU(out2, inplace=True),
            enn.PointwiseMaxPool(out2, 2)
        )

        self.block3 = enn.SequentialModule(
            enn.R2Conv(out2, out3, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(out3),
            enn.ReLU(out3, inplace=True),
            enn.PointwiseAdaptiveAvgPool(out3, output_size=1)
        )

        self.gpool = enn.GroupPooling(out3)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.block1.in_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gpool(x)
        x = x.tensor.view(x.tensor.size(0), -1)
        return self.fc(x)

# ============================================================
# Training and Testing
# ============================================================
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc=f"[Train] Epoch {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Train] Epoch {epoch} — Avg Loss: {total_loss / len(train_loader):.4f}")

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="[Test]"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"[Test] Accuracy: {acc:.2f}%")
    return acc

# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, transform=transform),
        batch_size=64, shuffle=False
    )

    model = SO2InvariantCNN(N=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Optionally load latest checkpoint
    checkpoints = sorted(
        [f for f in os.listdir("models") if f.startswith("so2_invariant_cnn_epoch")],
        key=lambda x: int(x.split("epoch")[1].split(".")[0])
    )

    start_epoch = 1
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Loading checkpoint: {latest}")
        model.load_state_dict(torch.load(f"models/{latest}", map_location=device))
        start_epoch = int(latest.split("epoch")[1].split(".")[0]) + 1

    for epoch in range(start_epoch, start_epoch + 3):
        train(model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, test_loader)

        save_path = f"models/so2_invariant_cnn_epoch{epoch}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path} (accuracy: {acc:.2f}%)")

if __name__ == "__main__":
    main()
