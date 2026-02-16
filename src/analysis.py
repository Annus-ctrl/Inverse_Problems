#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:25:41 2026

@author: ahaider
"""
# Reading fits files 

from tp_utils import * 
import matplotlib.pyplot as plt 
import numpy as np

# Read the staurn image
saturn = readfits("../data/saturn.fits")

# Read the PSF
psf = readfits("../data/saturn_psf.fits")

# Noramlizd the PSF

psf = normalize(psf)

# check that everything loaded perfectly 

print("Saturn shape", saturn.shape)
print ("Saturn_PSf", psf.shape)

# Display images 

figure()
imshow(saturn, cmap='gray')
colorbar()
plt.title("Observed Saturn HST")
show()


# Display PSF Image 
figure()
imshow(psf, cmap='gray')
colorbar()
plt.title("HST PSF")

show()

## Zero-pad PSF
psf_padded = zeropad(psf, saturn.shape) 

# Shift PSF so center is at (0,0)
psf_shifted = fftshift(psf_padded)

# FFTs
Y = fft(saturn)
H = fft(psf_shifted)

plt.figure(figsize=(5,5))
plt.imshow(np.log(abs2(H)+1e-10), cmap='gray')
plt.colorbar()
plt.title("PSF in Fourier space")
plt.show()

def wiener_filter(Y, H, mu):
    """
    Apply simplified Wiener filter in Fourier space.

    Inputs:
    Y  : FFT of observed image
    H  : FFT of PSF (padded)
    mu : regularization parameter

    Returns:
    x_rec : reconstructed image in real space
    """
    X_hat = conj(H) * Y / (abs2(H) + mu)   # Wiener filter formula
    x_rec = ifft(X_hat)                     # back to image space
    return x_rec
mu = 1e-2
x_rec = wiener_filter(Y, H, mu= 1e-2)
    
plt.figure(figsize=(5,5))
plt.imshow(x_rec,vmin = 0, cmap='gray')
plt.colorbar()
plt.title(f"Wiener filter reconstruction, μ = {mu}")
plt.show()

# How do we implement convolution operators?
#Forward model: 
#Hx
#Convolution is done via FFT:

wgt = np.ones(saturn.shape)
wgt[256, :] = 0      # mask the center
def W(x):
    return wgt * x

def Hx(x):
    return ifft(H * fft(x))

def HTx(x):
    return ifft(conj(H) * fft(x))

# Define the linear operator A(x) 
# A(x) = HT*W*H + mu*DT * D

def A(x):
    return HTx(W(Hx(x))) + mu * DtD(x)

#Define the right-hand side b
#b=HTy

b = HTx(W(saturn))
x0 = np.zeros_like(saturn)

x_cg = conjgrad (A=A, b=b, x=x0, maxiter = 50, grtol = 1e-5)

# Plots
plt.figure(figsize=(5,5))
plt.imshow(x_cg, vmin=0, cmap='gray')
plt.colorbar()
plt.title(f"Iterative reconstruction (CG), μ = {mu}")
plt.show()


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(x_rec, vmin=0, cmap='gray')
plt.title("Wiener")

plt.subplot(1,2,2)
plt.imshow(x_cg, vmin=0, cmap='gray')
plt.title("CG + regularization")
plt.show()

# Plot the Difference
diff = x_cg - x_rec
print("Max difference:", diff.max())
print("Mean difference:", diff.mean())

# Red = CG bigger than Wiener

#Blue = CG smaller than Wiener

# White ≈ no difference

diff = x_cg - x_rec

plt.figure(figsize=(6,6))
plt.imshow(diff, cmap='bwr', vmin=-50, vmax=50)  # bwr = blue-red diverging map
plt.colorbar()
plt.ylim(460, 50)
plt.title("Difference: CG - Wiener")
plt.show()

# checking symmetry 
# Random test vectors
u = np.random.randn(*saturn.shape)
v = np.random.randn(*saturn.shape)

lhs = np.sum(u * A(v))   # <u, A v>
rhs = np.sum(A(u) * v)   # <A u, v>

print("Symmetry check: |lhs - rhs| =", np.abs(lhs - rhs))

x = np.random.randn(*saturn.shape)
val = np.sum(x * A(x))
print("x^T A x =", val)

# Random missing pixels example
fraction_missing = 0.2  # 1% of pixels missing
wgt_masked = np.ones(saturn.shape)
wgt_masked[np.random.rand(*saturn.shape) < fraction_missing] = 0

# Redefine W and A with this masked weight
def W_masked(x):
    return wgt_masked * x

def A_masked(x):
    return HTx(W_masked(Hx(x))) + mu * DtD(x)

b_masked = HTx(W_masked(saturn))
x0 = np.zeros_like(saturn)

# Solve with CG
x_cg_masked = conjgrad(A=A_masked, b=b_masked, x=x0, maxiter=150, grtol=1e-5)

# Compare
diff_masked = x_cg_masked - x_cg
print("Max diff:", diff_masked.max())
print("Mean diff:", diff_masked.mean())

plt.figure(figsize=(6,6))                  # create a figure
plt.imshow(diff_masked, cmap='bwr', vmin=-50, vmax=50)  # bwr = blue-red diverging colormap
plt.colorbar()                              # add colorbar
plt.title("Difference: CG (masked) - CG (full)")  # optional title
plt.show()  
























