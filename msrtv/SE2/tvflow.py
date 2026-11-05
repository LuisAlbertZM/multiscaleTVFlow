"""
tvflow
======

Compute the SE(2) TV Flow, using PyTorch. The primary method is:
  1. `denoise`: perform roto-translational total variation (RT-TV) flow on
  SE(2) for denoising.[1][2]

References:
  [1]: A. Chambolle and Th. Pock.
    "Total roto-translational variation." In: Numerische Mathematik 142
    (2019), pp. 611--666.
    DOI:10.1007/s00211-019-01026-w.
  [2]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
    "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
    Positions and Orientations." In: Journal of Mathematical Imaging and
    Vision 63 (2021), pp. 237--262.
    DOI:10.1007/s10851-020-00991-4.
"""

import torch
import torch.nn.functional as F
from math import ceil, pi
from tqdm.auto import tqdm
from msrtv.utils import gaussian_blur_3d_separable, get_A_kernels, PSNR


def denoise(
    u: torch.Tensor,
    T: float,
    G_inv: tuple[float, float, float],
    σ_s: float,
    σ_o: float,
    ground_truth: torch.Tensor | None = None,
    max_val: float = 255.
) -> tuple[torch.Tensor, list[float]]:
    """
    Perform roto-translational total variation (RT-TV) flow[1] enhancing in
    SE(2), using an adaptation of the algorithm described by Smets et al.[2]

    Args:
        `u`: torch.Tensor initial condition, with shape [Or, H, W].
        `ξ`: float linking spatial dimension to orientational dimension.
        `T`: time that image is evolved under the DS PDE.
        `G_inv`: tuple[float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis.
        `σ_s`: standard deviation of the spatial regularisation, taking values
          greater than 0.
        `σ_o`: standard deviation of the orientational regularisation, taking
          values greater than 0.
        `ground_truth`: torch.Tensor ground truth, with shape [H, W].
        `max_val`: maximum value images take, for computing PSNR.

    Returns:
        torch.Tensor solution to the TV PDE with initial condition `u` at time
        `T`.

    References:
        [1]: A. Chambolle and Th. Pock.
          "Total roto-translational variation." In: Numerische Mathematik 142
          (2019), pp. 611--666.
          DOI:10.1007/s00211-019-01026-w.
        [2]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
          "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
          Positions and Orientations." In: Journal of Mathematical Imaging and
          Vision 63 (2021), pp. 237--262.
          DOI:10.1007/s10851-020-00991-4.
    """
    Or = u.shape[-3]
    dθ = 2 * pi / Or
    dt_opt = compute_timestep(1.0, dθ, G_inv)
    n = ceil(T / dt_opt)
    dt = T / n

    if ground_truth is not None:
        PSNRs = [
            PSNR(u.sum(-3), ground_truth, max_val=max_val)
        ]
    else:
        PSNRs = []
        
    for _ in tqdm(range(n)):
        u = step_TV(u, dt, G_inv, σ_s, σ_o)
        if ground_truth is not None:
            PSNRs.append(PSNR(u.sum(-3), ground_truth, max_val=max_val))

    return u, PSNRs


def step_TV(
    u: torch.Tensor,
    dt: float,
    G_inv: tuple[float, float, float],
    σ_s: float,
    σ_o: float,
) -> torch.Tensor:
    """
    Perform a single explicit step of roto-translational TV flow.

    The gradient is discretised with forward differences, and the divergence
    is defined as minus its adjoint, which uses backward differences.[1]

    Args:
        `u`: torch.Tensor current state, with shape [Or, H, W].
        `dt`: timestep, taking values greater than 0.
        `G_inv`: tuple[float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis.
        `σ_s`: standard deviation of the spatial regularisation of the
          gradient norm, taking values greater than 0.
        `σ_o`: standard deviation of the orientational regularisation of the
          gradient norm, taking values greater than 0.

    Returns:
        torch.Tensor state after one step, with shape [Or, H, W].

    References:
        [1]: A. Chambolle and Th. Pock.
          "A First-Order Primal-Dual Algorithm for Convex Problems with
          Applications to Imaging." In: Journal of Mathematical Imaging and
          Vision 40 (2011), pp. 120--145.
          DOI:10.1007/s10851-010-0251-1.
    """
    Or = u.shape[0]
    k_A1_grad, k_A2_grad = get_A_kernels(Or, u.device, True)
    k_A1_div, k_A2_div = get_A_kernels(Or, u.device, False)
    dθ = 2 * torch.pi / Or

    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    A1_u = F.conv2d(u_pad, k_A1_grad, groups=Or)
    A2_u = F.conv2d(u_pad, k_A2_grad, groups=Or)

    u_padded = F.pad(
        u[None, None],
        pad=(0, 0, 0, 0, 1, 1),
        mode="circular",
    )[0, 0]
    A3_u = (u_padded[2:] - u_padded[1:-1]) / dθ

    grad_norm = (G_inv[0] * A1_u**2 + G_inv[1] * A2_u**2 + G_inv[2] * A3_u**2).sqrt()
    grad_norm = gaussian_blur_3d_separable(grad_norm, (σ_s, σ_s, σ_o))  # + 1e-9

    normalised_A1_u = G_inv[0] * A1_u / grad_norm
    normalised_A2_u = G_inv[1] * A2_u / grad_norm
    normalised_A3_u = G_inv[2] * A3_u / grad_norm

    normalised_A1_u_pad = F.pad(normalised_A1_u, (1, 1, 1, 1), mode="constant")
    divergence_normalised_grad_1 = F.conv2d(normalised_A1_u_pad, k_A1_div, groups=Or)
    normalised_A2_u_pad = F.pad(normalised_A2_u, (1, 1, 1, 1), mode="constant")
    divergence_normalised_grad_2 = F.conv2d(normalised_A2_u_pad, k_A2_div, groups=Or)

    normalised_A3_u_padded = F.pad(
        normalised_A3_u[None, None],
        pad=(0, 0, 0, 0, 1, 1),
        mode="circular",
    )[0, 0]
    divergence_normalised_grad_3 = (normalised_A3_u_padded[1:-1] - normalised_A3_u_padded[:-2]) / dθ

    return u + dt * (
        divergence_normalised_grad_1
        + divergence_normalised_grad_2
        + divergence_normalised_grad_3
    )


def compute_timestep(
    dxy: float,
    dθ: float,
    G_inv: tuple[float, float, float],
):
    """
    Compute timestep to solve TV flow.

    `step_TV` is a forward Euler step of ∂u/∂t = Σ_i A_i (G_inv[i] A_i u / g),
    where g is the regularised gradient norm. Freezing g, this is linear
    diffusion with diffusivities G_inv[i] / g, for which we use the explicit
    diffusion timestep 2 / Σ_i (G_inv[i] / g) / h_i**2 with g = 1. This is a
    heuristic: since g vanishes in flat regions, no timestep is stable for all
    images, but the nonlinear scheme does not blow up when the bound is
    violated.

    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_inv`: tuple[float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis.

    Returns:
        timestep, taking values greater than 0.
    """
    return 2 / ((G_inv[0] + G_inv[1]) / dxy**2 + G_inv[2] / dθ**2)
