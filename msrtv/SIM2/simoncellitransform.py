"""
    simoncellitransform
    ===================

    Provides methods to compute the undecimated steerable wavelet transform of a
    2D image, using the filters of Simoncelli's steerable pyramid, as described
    in Simoncelli and Freeman "The Steerable Pyramid: A Flexible Architecture
    for Multi-Scale Derivative Computation" (1995). The primary methods are:
      1. `steerable_wavelet_stack`: compute the stack of steerable wavelets in
      the Fourier domain.
      2. `wavelet_transform`: compute the (real) wavelet transform of a 2D
      image with respect to the steerable wavelets.
      3. `inverse_wavelet_transform`: reconstruct a 2D image from its wavelet
      transform. Since the real parts of the wavelets form a tight frame, this
      is simply the adjoint of `wavelet_transform`.
"""

import torch
from math import log2
from functools import lru_cache


@lru_cache
def _polar_grids(
    N_y: int, N_x: int, device=None, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a grid of radial and angle coordinates in the Fourier domain, in
    cycles per pixel and in FFT ordering (i.e. the zero frequency is at index
    (0, 0)).

    For even image sizes, the Nyquist frequencies have no symmetric partner,
    which breaks the symmetry of the windows. We therefore fix the radius and
    angle at these frequencies, and also return a mask of them.
    """
    fys = torch.fft.fftfreq(N_y, device=device, dtype=dtype)[:, None]
    fxs = torch.fft.fftfreq(N_x, device=device, dtype=dtype)[None, :]
    ρs = torch.sqrt(fxs**2 + fys**2)
    θs = torch.atan2(fys, fxs)

    nyquist = (fxs.abs() == 0.5) | (fys.abs() == 0.5)
    ρs[nyquist] = 0.5 - 1e-9
    θs[nyquist] = 0.
    return ρs, θs, nyquist


def _radial_windows(
    ρs: torch.Tensor, scale: float, cut_freq: float, transition_width: float
) -> torch.Tensor:
    """
    Compute the high and low pass radial windows of Simoncelli's steerable
    pyramid, dilated by `scale`. The windows satisfy high**2 + low**2 = 1.
    """
    log_ρs = torch.log2(torch.clamp(ρs * scale, 0., 0.5 - 1e-9)) - log2(cut_freq)
    high = torch.cos(torch.clamp(log_ρs, -transition_width, 0.) * torch.pi / (2 * transition_width))
    low = torch.sqrt(1. - high**2)
    return high, low


def _angular_windows(θs: torch.Tensor, Nθ: int, nyquist: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular windows of the complex steerable pyramid, normalised
    such that the sum of their squares is 1. Each window is a single lobe
    cos(θ - θ_k)**order supported on |θ - θ_k| < π/2, so that, like the
    cakewavelets, the corresponding wavelets are complex: the real part is even
    (line detector) and the imaginary part is odd (edge detector). Hence, the
    orientations θ_k uniformly cover [0, 2π).

    The window of orientation θ_k is centred at frequency angle θ_k + π/2, so
    that the wavelet of orientation 0 is elongated along the width axis, and
    with increasing θ_k rotates towards the height axis.

    The order is chosen such that the windows have the same width relative to
    their spacing as in the real steerable pyramid with Nθ / 2 orientations.

    At the Nyquist frequencies, which are their own conjugate partner, the
    windows are symmetrised, such that the wavelets at θ_k and θ_k + π remain
    each other's complex conjugate.
    """
    order = max(Nθ // 2 - 1, 0)
    θs_k = 2 * torch.pi * torch.arange(Nθ, dtype=θs.dtype, device=θs.device) / Nθ + torch.pi / 2
    dθs = torch.remainder(θs[None, ...] - θs_k[:, None, None] + torch.pi,
                          2 * torch.pi) - torch.pi
    # The lobes are half-open, i.e. dθ ∈ [-π/2, π/2), such that frequencies on
    # their boundary belong to exactly one of the windows at θ_k and θ_k + π.
    # This only matters for order 0, i.e. Nθ < 4.
    tol = 1e-6
    support = (dθs >= -torch.pi / 2 - tol) & (dθs < torch.pi / 2 - tol)
    windows = torch.cos(dθs).clamp(min=0.)**order * support
    windows = windows / torch.sqrt((windows**2).sum(dim=0, keepdim=True) + 1e-12)

    windows_opposite = _angular_windows(θs + torch.pi, Nθ, torch.zeros_like(nyquist)) if nyquist.any() else windows
    return torch.where(nyquist, torch.sqrt((windows**2 + windows_opposite**2) / 2), windows)


def steerable_wavelet_stack(
    N_y: int, 
    N_x: int, 
    N_scales: int, 
    Nθ: int, 
    cut_freq: float = 0.25, 
    transition_width: float = 1., 
    dilation: float = 2.,
    device : torch.DeviceObjType | str | None = None,
    dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the steerable wavelets in the Fourier domain, in FFT ordering.

    Args:
        `N_y`, `N_x`: number of pixels in each spatial direction.
        `N_scales`: number of scales.
        `Nθ`: number of orientations, uniformly covering [0, 2π). Should be
          at least 2.
      Optional:
        `cut_freq`: frequency at which the finest high pass window reaches 1,
          in cycles per pixel. Defaults to 0.25.
        `transition_width`: width of the transition of the radial windows, in
          octaves. Smaller values give sharper partitions in the Fourier domain,
          and hence wavelets with more oscillations in the spatial domain.
          Defaults to 1.
        `dilation`: factor by which the wavelets are dilated in the spatial
          domain from one scale to the next. Defaults to 2.

    Returns:
        `filters`: band pass filters, of shape [N_scales, Nθ, N_y, N_x].
        `lowpass`: low pass residual filter, of shape [N_y, N_x].
    """
    ρs, θs, nyquist = _polar_grids(N_y, N_x, device=device, dtype=dtype)
    angular = _angular_windows(θs, Nθ, nyquist) * torch.where(nyquist, 1., 2**0.5)
    filters = torch.empty((N_scales, Nθ, N_y, N_x), device=device, dtype=dtype)
    lowpass = torch.ones_like(ρs)
    for s in range(N_scales):
        high, low = _radial_windows(ρs, dilation**s, cut_freq, transition_width)
        filters[s] = (lowpass * high)[None, ...] * angular
        lowpass = lowpass * low
    return filters, lowpass


def wavelet_transform(
    f: torch.Tensor, 
    filters: torch.Tensor, 
    lowpass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the wavelet transform of real image `f`, of shape [..., N_y, N_x],
    under the Fourier domain `filters` and `lowpass`, as computed by
    `steerable_wavelet_stack`. The band pass coefficients are real, of shape
    [..., N_scales, Nθ, N_y, N_x], and so is the low pass residual, of shape
    [..., N_y, N_x].
    """
    f_hat = torch.fft.fft2(f)
    U = torch.fft.ifft2(filters.to(f.device) * f_hat[..., None, None, :, :]).real
    U_low = torch.fft.ifft2(lowpass.to(f.device) * f_hat).real
    return U, U_low


def inverse_wavelet_transform(
    U: torch.Tensor, 
    U_low: torch.Tensor, 
    filters: torch.Tensor, 
    lowpass: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct a real image from its wavelet transform `U`, `U_low`, as
    computed by `wavelet_transform`.
    """
    f_hat = (filters.to(U.device) * torch.fft.fft2(U)).sum(dim=(-4, -3)) + lowpass.to(U.device) * torch.fft.fft2(U_low)
    return torch.fft.ifft2(f_hat).real
