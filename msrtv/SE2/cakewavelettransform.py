"""
cakewavelettransform
====================

Provides methods to compute the orientation score of a 2D image. The primary
methods are:
  1. `cakewavelet_stack`: compute the stack of cakewavelets, as described in
  Duits "Perceptual Organization in Image Analysis" (2005)
  (https://www.win.tue.nl/~rduits/THESISRDUITS.pdf).
  2. `wavelet_transform`: compute the wavelet transform of a 2D image, with
  respect to some wavelet (which need not be a cakewavelet).
"""

import torch
import torch.nn.functional as F
from math import floor, ceil, sqrt, factorial, comb, pi
from functools import lru_cache


@lru_cache
def cakewavelet_stack_fourier(
    N_spatial, orientations, spline_order=3, inflection_point=0.8, mn_order=8
):
    """
    Compute the cakewavelets in the Fourier domain.

    Parameters
    ----------
    N_spatial : int
        Spatial size of the lifting kernel in the Fourier domain will be
        `(N_spatial, N_spatial)`.

    orientations : int
        Number of orientations (dimension `Or`) to output.

    spline_order : int
        Order of the B-splines used to realize the kernel.

    inflection_point : float > 0 and < 1.
        Location of inflection point of the radial window of the cakewavelets.

    mn_order : int
        Order of the Taylor series of the radial window of the cakewavelets.

    Returns
    -------
    A tensor of shape `[Or, N_spatial, N_spatial]`
    """
    dtheta = 2 * pi / orientations
    rs_grid, thetas_grid = _polar_grids(N_spatial)
    thetas_grid = thetas_grid[None, ...]
    s = 2 * pi
    thetas = torch.arange(0, s, dtheta)[..., None, None]

    xs = _mod_offset(thetas_grid - thetas - pi / 2, 2 * pi, -pi) / dtheta
    cws_F = _B_spline(spline_order, xs)
    if inflection_point is not None:
        cws_F *= _radial_window(rs_grid, mn_order, inflection_point)[None, ...]

    return cws_F


def wavelet_transform(f: torch.Tensor, cws_F: torch.Tensor) -> torch.Tensor:
    """
    Return the wavelet transform of image f under the cakewavelets cws_F.

    Parameters
    ----------
    f : torch.Tensor
        Input tensor of shape `[..., H, W]`.

    cws_F : torch.Tensor
        Tensor of cakewavelets of shape `[Or, H_cws, W_cws]`, with
        `H_cws <= H` and `W_cws <= W`.

    Returns
    -------
    Tensor of shape `[..., Or, H, W]`.
    """
    H, W = f.shape[-2:]
    H_cws, W_cws = cws_F.shape[-2:]

    if (
        (H_cws, W_cws) != (H, W)
    ):  # Pad kernels so we can convolve by multiplication in Fourier domain.
        pad_1_l = floor((H - H_cws) / 2)
        pad_1_r = ceil((H - H_cws) / 2)
        pad_2_l = floor((W - W_cws) / 2)
        pad_2_r = ceil((W - W_cws) / 2)
        cws_F = F.pad(
            cws_F,
            (pad_2_r, pad_2_l, pad_1_r, pad_1_l),
            mode="constant",
            value=0.0,
        )

    cws_F = torch.fft.fftshift(cws_F, dim=(-2, -1)).to(f.device)

    f_hat = torch.fft.fft2(f)
    return torch.fft.ifft2(cws_F * f_hat[..., None, :, :]).real


def _mod_offset(x: torch.Tensor, period: float, offset: float) -> torch.Tensor:
    """
    Compute x modulo period with offset offset.

    Returns
    -------
    Tensor of shape `x.shape`.
    """
    return torch.remainder(x - offset, period) + offset


def _polar_grids(N_spatial: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a grid of radial and angle coordinates.

    Returns
    -------
    Tensor of shape `[N_spatial, N_spatial]`
    """
    centerx = ceil((N_spatial - 1) / 2)
    centery = centerx
    xs = torch.arange(N_spatial)
    ys = torch.arange(N_spatial)
    dxs = (xs - centerx)[None, :]
    dys = (ys - centery)[:, None]

    rs_grid = 2 * torch.sqrt(dxs**2 + dys**2) / N_spatial
    thetas_grid = torch.arctan2(dys, dxs)
    return rs_grid, thetas_grid


def _radial_window(
    rs_grid: torch.Tensor, n: int, inflection_point: float
) -> torch.Tensor:
    """
    Compute a smooth radial window in the Fourier domain for limiting the bandwidth of the cakewavelets.
    Corresponds to M_N, given by Eq. (4.41) in Duits "Perceptual Organization in Image Analysis" (2005).

    Returns
    -------
    Tensor of shape `rs_grid.shape`.
    """
    rho_matrix = rs_grid / sqrt(2 * inflection_point**2 / (1 + 2 * n))
    s = torch.zeros_like(rho_matrix)
    exp_rho_squared = torch.exp(-(rho_matrix**2))
    for k in range(n + 1):
        s = s + exp_rho_squared * rho_matrix ** (2 * k) / factorial(k)
    return s


def _B_spline(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Compute degree `n` B-splines.

    In this way, the sum of all cakewavelets in the Fourier domain is identically equal to 1 (within the disk M), while each cakewavelet varies smoothly in the angular direction in the Fourier domain.
    See Section 4.6 in Duits "Perceptual Organization in Image Analysis" (2005).

    Returns
    -------
    Tensor of shape `x.shape`
    """
    assert isinstance(n, int) and n >= 0, f"n must be a positive integer but got {n}."

    if n == 0:
        return 1.0 * (-1 / 2 <= x) * (x < 1 / 2)

    u = (n + 1) / 2 - x.abs()
    b = torch.zeros_like(u)
    for k in range(ceil((n + 1) / 2)):
        b += (-1) ** k * comb(n + 1, k) * (u - k).clamp(min=0) ** n
    return b / factorial(n)
