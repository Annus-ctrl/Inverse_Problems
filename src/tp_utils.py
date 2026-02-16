# -*- coding: utf-8 -*-
"""
To start your session, just do:

    from tp_utils import *          # utilities for the practical session

Work in progress ;-)
"""
# Insure compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

# Import modules using the similar conventions as in SciPy:
import numpy as _np              # to deal with numerical arrays
import matplotlib as _mpl        # for graphics
import matplotlib.pyplot as _plt # for plotting
import astropy.io.fits as _fits  # for reading/writing FITS files

# Provide an execfile function for Python ≥ 3.
# See https://stackoverflow.com/questions/1027714/how-to-execute-a-file-within-the-python-interpreter
import sys
if sys.version_info >= (3,0):
    def execfile(filename):
        """Execute the Python commands in a file."""
        exec(open(filename).read(), globals())

# Import some NumPy routines.
from numpy import conj, real, imag, log, sqrt
rand = _np.random.random_sample

# Other imports.
import math

# Import some routines for graphics.
figure = _plt.figure
clf = _plt.clf
cla = _plt.cla
fma = _plt.clf
imshow = _plt.imshow
close = _plt.close
show = _plt.show
get_cmap = _plt.get_cmap
colorbar = _plt.colorbar

# Turn interactive graphics on.
_plt.ion()

# Aliases/shortcuts for FFT routines.
fftshift = _np.fft.fftshift
ifftshift = _np.fft.ifftshift
fft = _np.fft.fftn
def ifft(x): return real(_np.fft.ifftn(x))

def abs2(x):
    """Returns the squared absolute value of its argument."""
    if _np.iscomplexobj(x):
        x_re = x.real
        x_im = x.imag
        return x_re*x_re + x_im*x_im
    else:
        return x*x

def normalize(psf):
    s = sum(psf.ravel())
    print("sum(PSF) = {0}".format(s))
    if s == 1.0: return psf
    if s > 0.0: return (1.0/s)*psf
    print('WARNING sum(PSF) = {0}'.format(s))
    return psf

def readfits(filename):
    """Read a FITS image from a file."""
    with _fits.open(filename) as hdus:
        return hdus[0].data

def writefits(filename, data, comment=None, history=None):
    """Write an image into a FITS file."""
    hdr = _fits.Header()
    if comment is not None:
        hdr['COMMENT'] = comment
    if history is not None:
        hdr['HISTORY'] = history
    hdu = _fits.PrimaryHDU(data, header=hdr)
    hdu.writeto(filename)

def conjgrad(A, b, x=None, inplace=False, maxiter=50, gatol=0.0, grtol=1e-5,
             quiet=False, viewer=None, restart=50):
    """Conjugate gradient routine.

    Solve the linear problem `A.x = b` by means the conjugate gradient method.

    Argument `A` is a function to compute the product of the "lef-hand-side
    (LHS) matrix" of the problem with a vector given in argument.

    Argument `b` is the "right-hand-side (RHS) vector".

    Optional argument `x0` is an initial solution to start with. By default,
    the initial solution is zero everywhere. If keyword `inplace` is `True`,
    the returned solution and the initial solution share the same array.

    Optional arguments `gatol` and `grtol` (0.0 and 1E-5 respectively by
    default) are the absolute and relative gradient tolerance for convergence.
    The conjugate gradient iterations are performed until the squared Euclidean
    norm of the residual is less or equal the largest between `gatol` and
    `grtol` times the squared Euclidean norm of the initial residuals.

    Optional argument `maxiter` (50 by default) is the maximum number of
    iterations to perform until returning the result. A warning message is
    printed if the number of iterations is exceeded (unless `quiet` is true).

    Optional argument `restart` (50 by default) is the maximum
    number of successive iterations to perform before restarting the conjugate
    gradient recurrence. Ignored if `restart < 1`.

    """

    def inner(x,y):
        """Compute inner product of X and Y regardless their shapes
        (their number of elements must however match)."""
        return _np.inner(x.ravel(),y.ravel())

    # Compute initial residuals.
    if x is None:
        r = _np.copy(b)
        x = _np.zeros(b.shape, dtype=b.dtype)
    else:
        if not inplace:
            x = _np.copy(x)
        r = b - A(x)
    rho = inner(r,r)
    epsilon = max(gatol, grtol*math.sqrt(rho))

    # Conjugate gradient iterations.
    beta = 0.0
    k = 0
    while True:
        if viewer is not None:
            viewer(x, k, math.sqrt(rho))
        if math.sqrt(rho) <= epsilon:
            break
        k += 1
        if k > maxiter:
            if not quiet:
                print("WARNING: too many iterations ({0})".format(maxiter))
            break

        # Next search direction.
        if beta == 0.0 or (restart > 0 and ((k - 1) % restart) == 0):
            p = r
        else:
            p = r + beta*p

        # Make optimal step along search direction.
        q = A(p)
        gamma = inner(p, q)
        if gamma <= 0.0:
            raise ValueError("Operator A is not positive definite")
        alpha = rho/gamma
        x += alpha*p
        r -= alpha*q
        rho_prev, rho = rho, inner(r,r)
        beta = rho/rho_prev
    return x

def zeropad(arr, shape):
    """Zero-pad array ARR to given shape.

    The contents of ARR is approximately centered in the result."""
    rank = arr.ndim
    if len(shape) != rank:
        raise ValueError("bad number of dimensions")
    diff = _np.asarray(shape) - _np.asarray(arr.shape)
    if diff.min() < 0:
        raise ValueError("output dimensions must be larger or equal input dimensions")
    offset = diff//2
    z = _np.zeros(shape, dtype=arr.dtype)
    if rank == 1:
        i0 = offset[0]; n0 = i0 + arr.shape[0]
        z[i0:n0] = arr
    elif rank == 2:
        i0 = offset[0]; n0 = i0 + arr.shape[0]
        i1 = offset[1]; n1 = i1 + arr.shape[1]
        z[i0:n0,i1:n1] = arr
    elif rank == 3:
        i0 = offset[0]; n0 = i0 + arr.shape[0]
        i1 = offset[1]; n1 = i1 + arr.shape[1]
        i2 = offset[2]; n2 = i2 + arr.shape[2]
        z[i0:n0,i1:n1,i2:n2] = arr
    elif rank == 4:
        i0 = offset[0]; n0 = i0 + arr.shape[0]
        i1 = offset[1]; n1 = i1 + arr.shape[1]
        i2 = offset[2]; n2 = i2 + arr.shape[2]
        i3 = offset[3]; n3 = i3 + arr.shape[3]
        z[i0:n0,i1:n1,i2:n2,i3:n3] = arr
    elif rank == 5:
        i0 = offset[0]; n0 = i0 + arr.shape[0]
        i1 = offset[1]; n1 = i1 + arr.shape[1]
        i2 = offset[2]; n2 = i2 + arr.shape[2]
        i3 = offset[3]; n3 = i3 + arr.shape[3]
        i4 = offset[4]; n4 = i4 + arr.shape[4]
        z[i0:n0,i1:n1,i2:n2,i3:n3,i4:n4] = arr
    elif rank == 6:
        i0 = offset[0]; n0 = i0 + arr.shape[0]
        i1 = offset[1]; n1 = i1 + arr.shape[1]
        i2 = offset[2]; n2 = i2 + arr.shape[2]
        i3 = offset[3]; n3 = i3 + arr.shape[3]
        i4 = offset[4]; n4 = i4 + arr.shape[4]
        i5 = offset[5]; n5 = i5 + arr.shape[5]
        z[i0:n0,i1:n1,i2:n2,i3:n3,i4:n4,i5:n5] = arr
    else:
        raise ValueError("too many dimensions")
    return z

def DtD(x):
    """Returns the result of D'⋅D⋅x where D is a (multi-dimensional)
    finite difference operator and D' is its transpose."""

    dims = x.shape
    r = _np.zeros(dims, dtype=x.dtype) # to store the result
    rank = x.ndim # number of dimensions
    if rank == 0: return r
    if dims[0] >= 2:
        dx = x[1:-1,...] - x[0:-2,...]
        r[1:-1,...] += dx
        r[0:-2,...] -= dx
    if rank == 1: return r
    if dims[1] >= 2:
        dx = x[:,1:-1,...] - x[:,0:-2,...]
        r[:,1:-1,...] += dx
        r[:,0:-2,...] -= dx
    if rank == 2: return r
    if dims[2] >= 2:
        dx = x[:,:,1:-1,...] - x[:,:,0:-2,...]
        r[:,:,1:-1,...] += dx
        r[:,:,0:-2,...] -= dx
    if rank == 3: return r
    if dims[3] >= 2:
        dx = x[:,:,:,1:-1,...] - x[:,:,:,0:-2,...]
        r[:,:,:,1:-1,...] += dx
        r[:,:,:,0:-2,...] -= dx
    if rank == 4: return r
    if dims[4] >= 2:
        dx = x[:,:,:,:,1:-1,...] - x[:,:,:,:,0:-2,...]
        r[:,:,:,:,1:-1,...] += dx
        r[:,:,:,:,0:-2,...] -= dx
    if rank == 5: return r
    raise ValueError("too many dimensions")


if __name__ == "__main__":
    print("testing TiPi!")
