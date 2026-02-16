# HST Saturn Image Deconvolution

This project reconstructs a blurred Hubble Space Telescope (HST) image of Saturn using the known Point Spread Function (PSF).

# Objective

To compare two approaches for solving the inverse imaging problem:

Wiener filtering (Fourier domain)

Conjugate Gradient (CG) with Tikhonov regularization

The forward model is:

y = Hx + n

where H is the PSF convolution operator.

# Methods

FFT-based convolution

Wiener filter:

X_hat(k) = H*(k) / (|H(k)|^2 + mu) * Y(k)

Iterative CG solution of:

(H^T W H + mu D^T D) x = H^T W y

Masked weights to simulate missing pixels

Finite-difference regularization

# Key Results

Wiener filtering is stable but smooths edges.

CG reconstruction preserves sharper structures (e.g., Saturn’s rings).

Strong masking suppresses high-frequency content.

Difference maps highlight sensitivity of edge regions to regularization.

# Tools

Python · NumPy · Astropy · Matplotlib · FFT · Conjugate Gradient

# Note

Some utility functions (FFT helpers and CG solver) were adapted from a practical session.
The reconstruction implementation, experiments, and analysis are my own work.
