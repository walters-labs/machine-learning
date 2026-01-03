import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse

device = 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# create the plots directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST(script_dir, train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(script_dir, train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

# -------------------------
# Model
# -------------------------
class SmallMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# Global adjacency matrix
# -------------------------
def global_adjacency_matrix(model):
    layers = [p for _, p in model.named_parameters() if p.ndim == 2]
    layer_sizes = [layers[0].shape[0]]
    for layer in layers:
        layer_sizes.append(layer.shape[1])
    total_nodes = sum(layer_sizes)
    A = torch.zeros((total_nodes, total_nodes), device=layers[0].device)

    offsets = [0]
    for size in layer_sizes[:-1]:
        offsets.append(offsets[-1] + size)

    for i, W in enumerate(layers):
        rows, cols = W.shape
        r0, c0 = offsets[i], offsets[i+1]
        A[r0:r0+rows, c0:c0+cols] = W
        A[c0:c0+cols, r0:r0+rows] = W.T
    return A

# -------------------------
# Path energy utilities
# -------------------------
def get_state_global(model):
    """Flatten full global adjacency matrix."""
    A = global_adjacency_matrix(model)
    return A.flatten()

def get_state_blocks(model):
    """Flatten block weights only (layerwise surrogate)."""
    states = []
    for _, W in model.named_parameters():
        if W.ndim == 2:
            states.append(W.flatten())
    return torch.cat(states)

def get_model_state(model, method):
    if method == "global":
        return get_state_global(model)
    elif method == "blocks":
        return get_state_blocks(model)
    else:
        raise ValueError(f"Unknown method: {method}")

# -------------------------
# Training with path energy regularization
# -------------------------
def train_and_eval(use_regularizer=False, mu=1e-4, epochs=10, method="blocks"):
    model = SmallMLP(hidden=128).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    acc_history, loss_history, path_energy_history = [], [], []

    # Initialize state for path energy
    prev_state = get_model_state(model, method).detach()

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, running_path_energy = 0.0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            # Path energy penalty
            curr_state = get_model_state(model, method)
            path_energy = torch.norm(curr_state - prev_state)**2
            if use_regularizer:
                loss += 0.5 * mu * path_energy
            running_path_energy += path_energy.item()
            prev_state = curr_state.detach()

            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        avg_path_energy = running_path_energy / len(train_loader.dataset)
        loss_history.append(avg_loss)
        path_energy_history.append(avg_path_energy)

        # Eval accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()
        acc = correct / total
        acc_history.append(acc)
        print(f"Epoch {epoch}, loss={avg_loss:.4f}, acc={acc:.4f}, path_energy={avg_path_energy:.6f}")

    return model, loss_history, acc_history, path_energy_history

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=1e-4, help="Regularization strength")
    parser.add_argument("--method", type=str, default="blocks", choices=["blocks", "global"],
                        help="How to compute path energy state")
    args = parser.parse_args()

    print("=== Baseline ===")
    base_model, base_loss, base_acc, base_pe = train_and_eval(use_regularizer=False, method=args.method)

    print("\n=== With Path Energy Regularizer ===")
    reg_model, reg_loss, reg_acc, reg_pe = train_and_eval(use_regularizer=True, mu=args.mu, method=args.method)

    # Plot results
    epochs = np.arange(1, len(base_acc)+1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, base_loss, label="Baseline")
    plt.plot(epochs, reg_loss, label=f"Path Reg (μ={args.mu})")
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.legend(); plt.title("Training Loss vs. Epoch")

    plt.subplot(1,2,2)
    plt.plot(epochs, base_acc, label="Baseline")
    plt.plot(epochs, reg_acc, label=f"Path Reg (μ={args.mu})")
    plt.xlabel("Epoch"); plt.ylabel("Test Accuracy")
    plt.legend(); plt.title("Test Accuracy vs. Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"loss_accuracy_plot_method={args.method}.png"), dpi=300)
    plt.show()

    # Path energy evolution
    plt.figure()
    plt.plot(base_pe, label="Baseline Path Energy")
    plt.plot(reg_pe, label="Reg Path Energy")
    plt.xlabel("Epoch"); plt.ylabel("Average Path Energy")
    plt.legend(); plt.title("Path energy per epoch")
    plt.savefig(os.path.join(plots_dir, f"path_energy_during_training_method={args.method}.png"), dpi=300)
    plt.show()

