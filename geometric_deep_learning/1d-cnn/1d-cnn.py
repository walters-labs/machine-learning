# translation_equivariance_demo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- 1. Define a simple 1D CNN ----
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, bias=False, padding=2)
        # Initialize kernel as a simple edge detector
        with torch.no_grad():
            self.conv1.weight[:] = torch.tensor([[[ -1, -0.5, 0, 0.5, 1 ]]])
    
    def forward(self, x):
        return self.conv1(x)

# ---- 2. Create an input signal ----
def make_signal(length=64, center=32, width=5):
    t = torch.arange(length, dtype=torch.float32)
    signal = torch.exp(-0.5 * ((t - center) / width) ** 2)
    return signal

# ---- 3. Utility: shift the signal ----
def shift_signal(x, shift):
    return torch.roll(x, shifts=shift, dims=-1)

# ---- 4. Run and visualize ----
if __name__ == "__main__":
    model = SimpleConvNet()
    model.eval()

    # Original and shifted inputs
    x = make_signal()
    x_shifted = shift_signal(x, shift=5)

    # Convert to batch form: [batch, channels, length]
    X = x.unsqueeze(0).unsqueeze(0)
    Xs = x_shifted.unsqueeze(0).unsqueeze(0)

    # Pass through CNN
    y = model(X).squeeze().detach()
    y_shifted = model(Xs).squeeze().detach()

    # Shift the original output to compare
    y_shifted_pred = shift_signal(y, shift=5)

    # ---- 5. Plot results ----
    plt.figure(figsize=(10,6))
    plt.subplot(3,1,1)
    plt.title("Input signals")
    plt.plot(x, label="original")
    plt.plot(x_shifted, label="shifted (+5)")
    plt.legend()

    plt.subplot(3,1,2)
    plt.title("CNN outputs")
    plt.plot(y, label="original output")
    plt.plot(y_shifted, label="output for shifted input")
    plt.legend()

    plt.subplot(3,1,3)
    plt.title("Output difference (should be near zero)")
    plt.plot(y_shifted_pred - y_shifted, label="difference after shift compensation")
    plt.legend()

    plt.tight_layout()
    plt.show()
