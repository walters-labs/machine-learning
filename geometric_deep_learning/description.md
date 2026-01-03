Geometric Deep Learning with e2cnn - Equivariant CNN Examples
==============================================================

This repository contains example code demonstrating the use of the
e2cnn library for building equivariant convolutional neural networks
(CNNs) on 1D and 2D data with symmetry groups (rotations).

---

Getting Started
---------------

1. Create a Python virtual environment (optional but recommended):

   On macOS/Linux:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:

   ```
   pip install torch torchvision e2cnn matplotlib numpy pillow
   ```

   Make sure you have Python 3.7+ installed.

---

## Files & Models
--------------

- `1d_equivariant_cnn.py`  
  A simple 1D CNN equivariant to discrete cyclic groups, demonstrating
  equivariance in one dimension.

- `2d-SO_2-cnn.py`  
  A 2D CNN equivariant to the discrete SO(2) rotation group (rotations
  by multiples of 90°). Contains:

  - An equivariant convolutional network using e2cnn.
  - A test function validating equivariance by comparing outputs for
    rotated inputs vs. rotated outputs.

---

Running the Equivariance Test
-----------------------------

Run the 2D SO(2) equivariance demo with:

```
python 2d-SO_2-cnn.py
```

This will:

- Generate a test image (a vertical bar),
- Rotate the input image by 90°,
- Pass both images through the equivariant model,
- Compare the output tensors under the group action,
- Print maximum and mean absolute differences,
- Plot side-by-side visualizations of outputs.

You should see a message:

```
Equivariance test passed (within numerical tolerance).
```

---

Notes
-----

- The test validates architectural equivariance, independent of training.
- A warning about deprecated PyTorch indexing may appear; this can be safely ignored.
- The code runs on CPU by default but can be adapted for GPU.
- To achieve invariance rather than equivariance, consider adding
  an invariant pooling layer after the equivariant layers.

---

Feel free to extend or integrate these examples into your projects!

---

MIT License
© 2025 Jackson Walters. All rights reserved. 
Date: November 2025
