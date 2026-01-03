"""
2d-SO2-invariant-cnn.py

A stronger, cleaner example of a 2D rotation-invariant CNN using e2cnn.

- Builds an equivariant feature extractor (regular representations).
- Applies GroupPooling to achieve rotation-invariance.
- Global average pooling + linear head for compact embeddings.
- Includes stronger invariance tests across multiple angles and with noise.
- Trains briefly on synthetic data to demonstrate the training process.

Requirements:
    pip install torch torchvision e2cnn matplotlib numpy pillow

Run:
    python 2d-SO2-invariant-cnn.py --viz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from e2cnn import gspaces
from e2cnn import nn as enn
from PIL import Image
import numpy as np
import argparse


# ---------------------------------------------------------------------
# Utility: simple bar image generator + rotation
# ---------------------------------------------------------------------
def make_bar_image(size=64, bar_width=8):
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half = bar_width // 2
    img[:, center - half:center + half] = 1.0
    return img


def pil_rotate(arr, angle):
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil_r = pil.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return np.array(pil_r).astype(np.float32) / 255.0


# ---------------------------------------------------------------------
# Model: equivariant + invariant CNN
# ---------------------------------------------------------------------
class InvariantEquivariantNet(nn.Module):
    def __init__(self, N=4, n_classes=3):
        super().__init__()
        self.N = N
        self.gspace = gspaces.Rot2dOnR2(N=N)

        in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        out1 = enn.FieldType(self.gspace, [self.gspace.regular_repr])
        out2 = enn.FieldType(self.gspace, [self.gspace.regular_repr])

        self.block1 = enn.R2Conv(in_type, out1, kernel_size=7, padding=3, bias=False)
        self.relu1 = enn.ReLU(out1, inplace=True)
        self.block2 = enn.R2Conv(out1, out2, kernel_size=5, padding=2, bias=False)
        self.relu2 = enn.ReLU(out2, inplace=True)

        self.gpool = enn.GroupPooling(out2)
        self.fc = nn.Linear(1, n_classes)

        self.in_type = in_type
        self.out_type = out2

    def forward(self, geo_x):
        x = self.relu1(self.block1(geo_x))
        x = self.relu2(self.block2(x))
        x = self.gpool(x)
        t = x.tensor.mean(dim=[2, 3])
        return self.fc(t)


# ---------------------------------------------------------------------
# Invariance tests
# ---------------------------------------------------------------------
def test_invariance(model, angles=[45, 90, 135, 180], noise=0.02):
    device = next(model.parameters()).device
    model.eval()

    img = make_bar_image(size=64, bar_width=10)
    img += noise * np.random.randn(*img.shape)
    img = np.clip(img, 0, 1)

    X = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    geo_X = enn.GeometricTensor(X, model.in_type)
    base_out = model(geo_X).detach()

    print("\n=== Invariance Tests ===")
    tol = 5e-3 if model.N > 0 else 1e-4  # discrete vs continuous
    for angle in angles:
        img_r = pil_rotate(img, angle=angle)
        Xr = torch.from_numpy(img_r).unsqueeze(0).unsqueeze(0).to(device)
        geo_Xr = enn.GeometricTensor(Xr, model.in_type)
        out_r = model(geo_Xr).detach()

        diff = (out_r - base_out).abs()
        passed = diff.max() < tol
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"{color}Rotation {angle:>3}° | max: {diff.max():.3e} | mean: {diff.mean():.3e}{reset}")
        if not passed:
            print(f"⚠️  Warning: Invariance deviation exceeds tolerance ({tol}).")

    print("✅ Invariance test completed.")


# ---------------------------------------------------------------------
# Demonstration training loop
# ---------------------------------------------------------------------
def train_demo(model, epochs=3, lr=1e-2):
    print("\n=== Training Demo ===")
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Random synthetic inputs
        imgs = [make_bar_image(size=64, bar_width=np.random.randint(4, 12)) for _ in range(8)]
        X = torch.tensor(np.stack(imgs))[:, None, :, :].float().to(device)
        y = torch.randn(8, 3).to(device)

        geo_X = enn.GeometricTensor(X, model.in_type)
        out = model(geo_X)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()
        print(f"[Train] Epoch {epoch}/{epochs} — Loss: {loss.item():.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SO(2) invariant CNN demo (e2cnn).")
    parser.add_argument("--N", type=int, default=4, help="Number of discrete rotations")
    parser.add_argument("--viz", action="store_true", help="Show input visualization")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = InvariantEquivariantNet(N=args.N).to(device)

    # Train demo
    train_demo(model, epochs=3, lr=1e-2)

    # Invariance tests
    test_invariance(model, angles=[45, 90, 135, 180], noise=0.01)

    if args.viz:
        img = make_bar_image()
        img_r = pil_rotate(img, 90)
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Rotated 90°")
        plt.imshow(img_r, cmap="gray")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
