"""
tvflow
======

Compute the SE(2) TV Flow, using PyTorch.
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
) -> torch.Tensor:
    """
    Perform TV flow enhancing in SE(2), using an adaptation of the algorithm
    described by Smets et al.[1]

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
        [1]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
          "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
          Positions and Orientations". In: Journal of Mathematical Imaging and
          Vision 63 (2021), pp. 237--262.
          DOI:10.1007/s10851-020-00991-4.
    """
    Or = u.shape[-3]
    dθ = 2 * pi / (Or - 1)
    dt_opt = compute_timestep(1.0, dθ, G_inv)
    n = ceil(T / dt_opt)
    dt = T / n

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
    Or = u.shape[0]
    k_A1, k_A2 = get_A_kernels(Or, u.device)
    dθ = 2 * torch.pi / (Or - 1)

    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    A1_u = F.conv2d(u_pad, k_A1, groups=Or)
    A2_u = F.conv2d(u_pad, k_A2, groups=Or)

    u_padded = F.pad(
        u[None, None],
        pad=(0, 0, 0, 0, 1, 1),
        mode="circular",
    ).squeeze()
    A3_u = (u_padded[2:] - u_padded[:-2]) / (2.0 * dθ)
    #print(A1_u.max(), A2_u.max(), A3_u.max())

    grad_norm = (G_inv[0] * A1_u**2 + G_inv[1] * A2_u**2 + G_inv[2] * A3_u**2).sqrt()
    grad_norm = gaussian_blur_3d_separable(grad_norm, (σ_s, σ_s, σ_o))  # + 1e-9

    normalised_A1_u = G_inv[0] * A1_u / grad_norm
    normalised_A2_u = G_inv[1] * A2_u / grad_norm
    normalised_A3_u = G_inv[2] * A3_u / grad_norm

    normalised_A1_u_pad = F.pad(normalised_A1_u, (1, 1, 1, 1), mode="replicate")
    divergence_normalised_grad_1 = F.conv2d(normalised_A1_u_pad, k_A1, groups=Or)
    normalised_A2_u_pad = F.pad(normalised_A2_u, (1, 1, 1, 1), mode="replicate")
    divergence_normalised_grad_2 = F.conv2d(normalised_A2_u_pad, k_A2, groups=Or)

    normalised_A3_u_padded = F.pad(
        normalised_A3_u[None, None],
        pad=(0, 0, 0, 0, 1, 1),
        mode="circular",
    ).squeeze()
    divergence_normalised_grad_3 = (
        normalised_A3_u_padded[2:] - normalised_A3_u_padded[:-2]
    ) / (2.0 * dθ)

    return u + dt * (
        divergence_normalised_grad_1
        + divergence_normalised_grad_2
        + divergence_normalised_grad_3
    )


def compute_timestep(dxy, dθ, G_inv):
    """
    Compute timestep to solve TV flow.

    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_inv`: tuple[float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis.

    Returns:
        timestep, taking values greater than 0.
    """
    return dxy**2 * dθ / (2 * ((G_inv[0] + G_inv[1]) * dxy + G_inv[2] * dθ))
