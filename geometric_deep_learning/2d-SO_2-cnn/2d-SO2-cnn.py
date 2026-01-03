"""
e2cnn_equivariance_demo.py
Demonstrates discrete SO(2) (rotations by 90°) equivariance using e2cnn.

Requirements:
    - torch, torchvision
    - e2cnn

Run:
    python e2cnn_equivariance_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# e2cnn imports
from e2cnn import gspaces
from e2cnn import nn as enn

# torchvision for image rotation (PIL-based)
from torchvision import transforms
from PIL import Image
import numpy as np


# -------------------------
# 1) Model: a single equivariant conv
# -------------------------
class SmallEquivariantNet(nn.Module):
    def __init__(self, N=4):
        """
        N: number of discrete rotations (use 4 -> 0,90,180,270 degrees)
        """
        super().__init__()

        # Define the rotation group on the plane
        self.gspace = gspaces.Rot2dOnR2(N=N)

        # Input is a scalar field (grayscale image) -> trivial representation
        in_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr])

        # Output: one *regular* field (regular repr has dimension N),
        # so the output tensor will have `N` channels internally.
        out_type = enn.FieldType(self.gspace, [self.gspace.regular_repr])

        # Equivariant convolution (replaces Conv2d)
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=7, padding=3, bias=False)

        # a nonlinearity that is compatible with e2cnn (pointwise)
        self.relu = enn.ReLU(out_type, inplace=True)

        # optional second equivariant conv to show stacking works
        mid_type = out_type
        out_type2 = enn.FieldType(self.gspace, [self.gspace.regular_repr])
        self.conv2 = enn.R2Conv(mid_type, out_type2, kernel_size=5, padding=2, bias=False)
        self.relu2 = enn.ReLU(out_type2, inplace=True)

        # store types for external use
        self.in_type = in_type
        self.out_type = out_type2

    def forward(self, x):
        # x will be a GeometricTensor (see usage below)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


# -------------------------
# 2) Utilities: construct an image & rotate it
# -------------------------
def make_bar_image(size=64, bar_width=8):
    """
    Create a centered vertical bar on a zero background (shape HxW).
    """
    img = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half = bar_width // 2
    img[:, center - half : center + half] = 1.0
    return img


def pil_rotate(arr, angle):
    """Rotate numpy array (HxW) by angle degrees using PIL (keeps same size)."""
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil_r = pil.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    arr_r = np.array(pil_r).astype(np.float32) / 255.0
    return arr_r


# -------------------------
# 3) Equivariance test (N = 4 -> rotation by 90 deg)
# -------------------------
def test_equivariance():
    device = torch.device("cpu")
    N = 4  # 4 discrete rotations (0,90,180,270)
    angle = 90  # test rotation

    # build model
    model = SmallEquivariantNet(N=N).to(device)
    model.eval()

    # create input image
    img = make_bar_image(size=64, bar_width=10)  # H x W, values 0..1
    img_rot = pil_rotate(img, angle=angle)

    # visualize inputs quickly
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.title("original"); plt.imshow(img, cmap="gray"); plt.axis("off")
    plt.subplot(1,2,2); plt.title(f"rotated {angle}°"); plt.imshow(img_rot, cmap="gray"); plt.axis("off")
    plt.suptitle("Inputs")
    plt.show()

    # Convert to torch tensors shaped [B, C, H, W]
    X = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)      # [1,1,H,W]
    Xr = torch.from_numpy(img_rot).unsqueeze(0).unsqueeze(0).to(device) # rotated input

    # Wrap into GeometricTensor with the input field type (scalar)
    geo_X = enn.GeometricTensor(X, model.in_type)
    geo_Xr = enn.GeometricTensor(Xr, model.in_type)

    # Forward pass
    out = model(geo_X)   # GeometricTensor with regular repr field
    outr = model(geo_Xr)

    # Extract raw tensors [B, C, H, W]
    Y = out.tensor.detach()
    Yr = outr.tensor.detach()

    # For N=4 regular_repr, the channel dimension corresponds to the N orientations
    # The group action of a 90-degree rotation is:
    #   1) cyclically shift the orientation channels by +1
    #   2) rotate the spatial grid by 90 degrees
    #
    # So simulate applying the group element to Y:
    # (a) channel permutation (roll by +1 on dim=1)
    Y_perm = torch.roll(Y, shifts=1, dims=1)  # cyclic shift channels
    # (b) spatial rotation by 90 deg (k=1)
    # torch.rot90 rotates in plane dims (H,W) which are dims 2 and 3
    Y_transformed = torch.rot90(Y_perm, k=1, dims=[2, 3])

    # Now compare Y_transformed (original output acted on by rotation) to Yr (output of rotated input)
    diff = (Yr - Y_transformed)
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    print(f"Max abs difference after group action: {max_abs:.3e}")
    print(f"Mean abs difference after group action: {mean_abs:.3e}")

    # Visual check: plot first few channels of Yr and Y_transformed side-by-side
    nch = Y.shape[1]
    nshow = min(nch, 4)
    plt.figure(figsize=(10, 4))
    for i in range(nshow):
        plt.subplot(2, nshow, i+1)
        plt.imshow(Y_transformed[0, i].cpu(), cmap="RdBu", vmin=-1, vmax=1)
        plt.title(f"Y_transf ch {i}")
        plt.axis("off")

        plt.subplot(2, nshow, nshow + i+1)
        plt.imshow(Yr[0, i].cpu(), cmap="RdBu", vmin=-1, vmax=1)
        plt.title(f"Yr (rot input) ch {i}")
        plt.axis("off")
    plt.suptitle("Top: transformed original output | Bottom: output for rotated input")
    plt.show()

    # Assert near-equality
    assert max_abs < 1e-5, "Equivariance test failed (difference too large)"
    print("Equivariance test passed (within numerical tolerance).")


if __name__ == "__main__":
    test_equivariance()
