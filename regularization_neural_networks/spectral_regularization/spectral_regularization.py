import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
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
    layers = [p for n, p in model.named_parameters() if p.ndim == 2]
    layer_sizes = [layers[0].shape[0]]
    for layer in layers:
        layer_sizes.append(layer.shape[1])
    total_nodes = sum(layer_sizes)
    A = torch.zeros((total_nodes, total_nodes), device=layers[0].device)

    layer_offsets = [0]
    for size in layer_sizes[:-1]:
        layer_offsets.append(layer_offsets[-1] + size)

    for i, W in enumerate(layers):
        rows, cols = W.shape
        row_start = layer_offsets[i]
        col_start = layer_offsets[i + 1]
        A[row_start:row_start + rows, col_start:col_start + cols] = W
        A[col_start:col_start + cols, row_start:row_start + rows] = W.T
    return A

# -------------------------
# Spectral energy (single layer)
# -------------------------
def spectral_energy(W):
    if W.ndim != 2:
        return torch.tensor(0.0, device=W.device)
    if W.size(0) == W.size(1):
        S = 0.5 * (W + W.t())
        eigs = torch.linalg.eigvalsh(S)
        return torch.sum(torch.abs(eigs))
    else:
        S = W.t() @ W if W.size(1) < W.size(0) else W @ W.t()
        eigs = torch.linalg.eigvalsh(S)
        return torch.sum(torch.sqrt(torch.clamp(eigs, min=1e-12)))

# -------------------------
# Global spectral energy
# -------------------------
def spectral_energy_global(model, method='layerwise', monitor=False):
    total = 0.0
    for _, W in model.named_parameters():
        if W.ndim == 2:
            if method == 'layerwise':
                val = spectral_energy(W)
            elif method == 'nuclear_norm':
                sv = torch.linalg.svdvals(W)
                val = 2 * sv.sum()
            elif method == 'frobenius_norm':
                val = torch.sum(W**2)
            elif method == 'global_adjacency':
                A = global_adjacency_matrix(model)
                eigs = torch.linalg.eigvalsh(A)
                val = torch.sum(torch.abs(eigs))
            else:
                raise ValueError(f"Unknown method {method}")
            total = total + val

    # Differentiable tensor for training
    if not monitor:
        return total
    # Scalar float for logging/plotting
    else:
        return total.detach().cpu().item()

# -------------------------
# Training loop
# -------------------------
def train_and_eval(use_regularizer=False, mu=1e-4, epochs=10, method='layerwise'):
    model = SmallMLP(hidden=128).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    acc_history = []
    loss_history = []
    spectral_history = []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if use_regularizer:
                spectral_loss = spectral_energy_global(model, method=method, monitor=False)
                loss += mu * spectral_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

        # Log spectral energy (non-differentiable, safe)
        spectral_history.append(spectral_energy_global(model, method=method, monitor=True))

        # Eval accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds==y).sum().item()
                total += y.numel()
        acc = correct/total
        acc_history.append(acc)
        print(f"Epoch {epoch}, loss={avg_loss:.4f}, acc={acc:.4f}, spectral_energy={spectral_history[-1]:.4f}")

    return model, loss_history, acc_history, spectral_history

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with spectral regularization")
    parser.add_argument("--mu", type=float, default=1e-4, help="Regularization strength")
    parser.add_argument("--method", type=str, default='layerwise', help="Regularization type")
    args = parser.parse_args()

    print(f"Using mu = {args.mu}")
    print(f"Using method = {args.method}")

    print("=== Baseline ===")
    baseline_model, baseline_loss, baseline_acc, baseline_spec = train_and_eval(use_regularizer=False, method=args.method)

    print("\n=== With Spectral Energy Regularizer ===")
    reg_model, reg_loss, reg_acc, reg_spec = train_and_eval(use_regularizer=True, mu=args.mu, method=args.method)

    epochs = np.arange(1, len(baseline_acc)+1)

    # Loss
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, baseline_loss, label="Baseline")
    plt.plot(epochs, reg_loss, label=f"Spectral Reg (μ={args.mu})")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.title("Training Loss vs. Epoch")

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, baseline_acc, label="Baseline")
    plt.plot(epochs, reg_acc, label=f"Spectral Reg (μ={args.mu})")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.title("Test Accuracy vs. Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"loss_accuracy_plot_method={args.method}.png"), dpi=300)
    plt.show()

    # Layer spectra
    def layer_spectra(model):
        spectra = {}
        for name, p in model.named_parameters():
            if p.ndim == 2:
                with torch.no_grad():
                    if p.size(0) == p.size(1):
                        S = 0.5*(p + p.t())
                    else:
                        S = p.t() @ p if p.size(1) < p.size(0) else p @ p.t()
                    eigs = torch.linalg.eigvalsh(S.cpu())
                    spectra[name] = eigs.numpy()
        return spectra

    baseline_spectra = layer_spectra(baseline_model)
    reg_spectra      = layer_spectra(reg_model)

    layers = list(baseline_spectra.keys())
    num_layers = len(layers)
    cols = 2
    rows = math.ceil(num_layers / cols)

    plt.figure(figsize=(cols*6, rows*4))
    for i, layer in enumerate(layers, 1):
        plt.subplot(rows, cols, i)
        plt.hist(baseline_spectra[layer], bins=50, alpha=0.5, label="Baseline")
        plt.hist(reg_spectra[layer], bins=50, alpha=0.5, label="Spectral Reg")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Count")
        plt.legend()
        plt.title(f"Eigenvalue spectrum: {layer}")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"eigenvalue_distribution_by_layer_method={args.method}.png"), dpi=300)
    plt.show()

    # Spectral energy over epochs
    plt.figure(figsize=(12,6))
    plt.plot(baseline_spec, label="Baseline")
    plt.plot(reg_spec, label="w/ spectral regularization")
    plt.xlabel("Epoch")
    plt.ylabel("Spectral Energy")
    plt.title("Spectral energy over epochs")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"spectral_energy_during_training_method={args.method}.png"), dpi=300)
    plt.show()
