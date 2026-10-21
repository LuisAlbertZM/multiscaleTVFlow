"""
tvflow
======

Compute the SIM(2) TV Flow, using PyTorch.
"""

import torch
import torch.nn.functional as F
from math import ceil, pi
from tqdm.auto import tqdm
from msrtv.utils import gaussian_blur_4d_separable, get_A_kernels, PSNR
from msrtv.SIM2.simoncellitransform import inverse_wavelet_transform
from msrtv.SIM2.utils import GroundTruth


def denoise(
    u: torch.Tensor,
    T: float,
    G_inv: tuple[float, float, float, float],
    a: float,
    σ_s: float,
    σ_o: float,
    σ_a: float,
    ground_truth: GroundTruth | None = None,
    max_val: float = 255.
) -> tuple[torch.Tensor, list[float]]:
    """
    Perform TV flow enhancing in SIM(2), using an adaptation of the algorithm
    described by Smets et al.[1]

    Args:
        `u`: torch.Tensor initial condition, with shape [S, Or, H, W].
        `ξ`: float linking spatial dimension to orientational dimension.
        `T`: time that image is evolved under the DS PDE.
        `G_inv`: tuple[float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis.
        `a`: scale something
        `σ_s`: standard deviation of the spatial regularisation, taking values
          greater than 0.
        `σ_o`: standard deviation of the orientational regularisation, taking
          values greater than 0.
        `σ_a`: standard deviation of the scale regularisation, taking
          values greater than 0.
        `ground_truth`: GroundTruth instance for computing PSNR.
          groundtruth: torch.Tensor ground truth, with shape [H, W].
          u_low: torch.Tensor low passed noisy image, with shape [H, W].
          filters: torch.Tensor filters used for lifting.
          lowpass: torch.Tensor lowpass filter used for lifting.
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
    S = u.shape[-4]
    Or = u.shape[-3]
    dθ = 2 * pi / Or
    dt_opt = compute_timestep(1.0, dθ, G_inv, a, S)
    n = ceil(T / dt_opt)
    dt = T / n

    PSNRs = []
    for _ in tqdm(range(n)):
        u = step_TV(u, dt, G_inv, a, σ_s, σ_o, σ_a)
        if ground_truth is not None:
            PSNRs.append(PSNR(
                inverse_wavelet_transform(u, ground_truth.u_low, ground_truth.filters, ground_truth.lowpass),
                ground_truth.ground_truth,
                max_val=max_val
            ))

    return u, PSNRs


def step_TV(
    u: torch.Tensor, 
    dt: float,
    G_inv: tuple[float, float, float, float],
    a: float,
    σ_s: float,
    σ_o: float,
    σ_a: float,
) -> torch.Tensor:
    S, Or, H, W = u.shape
    k_A1, k_A2 = get_A_kernels(Or, u.device)
    dθ = 2 * torch.pi / Or
    scales = (a ** torch.arange(S, device=u.device))

    u_reshaped = u.reshape(1, S * Or, H, W)
    u_pad = F.pad(u_reshaped, (1, 1, 1, 1), mode="replicate")
    A1_u = F.conv2d(u_pad, k_A1.repeat(S, 1, 1, 1), groups=S * Or)
    A2_u = F.conv2d(u_pad, k_A2.repeat(S, 1, 1, 1), groups=S * Or)

    A1_u = scales.view(S,1,1,1) * A1_u.reshape(S, Or, H, W)
    A2_u = scales.view(S,1,1,1) * A2_u.reshape(S, Or, H, W)

    u_padded = torch.cat(
        [u[:, -1:], u, u[:, :1]],
        dim=1,
    )
    A3_u = (u_padded[:, 2:] - u_padded[:, :-2]) / (2.0 * dθ)

    dscale_forward = a - 1.
    dscale_backward = 1. - 1. / a
    A4_forward = (u[1:] - u[:-1]) / dscale_forward
    A4_backward = (u[1:] - u[:-1]) / dscale_backward

    A4_u = torch.zeros_like(u)

    A4_u[1:-1] = (A4_forward[1:] + A4_backward[:-1]) / 2
    A4_u[0]    = A4_forward[0]
    A4_u[-1]   = A4_backward[-1]

    grad_norm = (G_inv[0] * A1_u**2 + G_inv[1] * A2_u**2 + G_inv[2] * A3_u**2 + G_inv[3] * A4_u**2).sqrt()
    grad_norm = gaussian_blur_4d_separable(grad_norm, (σ_s, σ_s, σ_o, σ_a))  # + 1e-9

    normalised_A1_u = G_inv[0] * A1_u / grad_norm
    normalised_A2_u = G_inv[1] * A2_u / grad_norm
    normalised_A3_u = G_inv[2] * A3_u / grad_norm
    normalised_A4_u = G_inv[3] * A4_u / grad_norm

    normalised_A1_u_pad = F.pad(normalised_A1_u.reshape(1, S * Or, H, W), (1, 1, 1, 1), mode="replicate")
    divergence_normalised_grad_1 =  F.conv2d(normalised_A1_u_pad, k_A1.repeat(S, 1, 1, 1), groups=S * Or)
    divergence_normalised_grad_1 = scales.view(S,1,1,1) * divergence_normalised_grad_1.reshape(S, Or, H, W)

    normalised_A2_u_pad = F.pad(normalised_A2_u.reshape(1, S * Or, H, W), (1, 1, 1, 1), mode="replicate")
    divergence_normalised_grad_2 = F.conv2d(normalised_A2_u_pad, k_A2.repeat(S, 1, 1, 1), groups=S * Or)
    divergence_normalised_grad_2 = scales.view(S,1,1,1) * divergence_normalised_grad_2.reshape(S, Or, H, W)

    normalised_A3_u_padded = torch.cat(
        [
            normalised_A3_u[:, -1:],
            normalised_A3_u,
            normalised_A3_u[:, :1],
        ],
        dim=1,
    )
    divergence_normalised_grad_3 = (
        normalised_A3_u_padded[:, 2:] - normalised_A3_u_padded[:, :-2]
    ) / (2.0 * dθ)

    divergence_normalised_grad_4 = -2 * normalised_A4_u
    A4_forward_normalised_A4_u = (normalised_A4_u[1:] - normalised_A4_u[:-1]) / dscale_forward
    A4_backward_normalised_A4_u = (normalised_A4_u[1:] - normalised_A4_u[:-1]) / dscale_backward

    divergence_normalised_grad_4[1:-1] += (A4_forward_normalised_A4_u[1:] + A4_backward_normalised_A4_u[:-1]) / 2
    divergence_normalised_grad_4[0]    += A4_forward_normalised_A4_u[0]
    divergence_normalised_grad_4[-1]   += A4_backward_normalised_A4_u[-1]

    return u + dt * (
        divergence_normalised_grad_1
        + divergence_normalised_grad_2
        + divergence_normalised_grad_3
        + divergence_normalised_grad_4
        
    )

def _scale_operator_norm(a: float, S: int) -> float:
    """
    Compute the spectral norm of the linear operator in the scale direction
    of `step_TV`, i.e. v -> -(A4 v - 2 v) composed with A4, where A4 is the
    scale derivative with forward step a - 1, backward step 1 - 1/a, and
    one-sided differences at the boundaries.
    """
    if S < 2:
        return 0.
    dscale_forward = a - 1.
    dscale_backward = 1. - 1. / a
    A4 = torch.zeros((S, S), dtype=torch.float64)
    A4[0, 0], A4[0, 1] = -1. / dscale_forward, 1. / dscale_forward
    A4[-1, -2], A4[-1, -1] = -1. / dscale_backward, 1. / dscale_backward
    for s in range(1, S - 1):
        A4[s, s + 1] += 1. / (2 * dscale_forward)
        A4[s, s] += 1. / (2 * dscale_backward) - 1. / (2 * dscale_forward)
        A4[s, s - 1] -= 1. / (2 * dscale_backward)
    divergence_4 = A4 - 2 * torch.eye(S, dtype=torch.float64)
    return torch.linalg.matrix_norm(divergence_4 @ A4, ord=2).item()


def compute_timestep(
    dxy: float,
    dθ: float,
    G_inv: tuple[float, float, float, float],
    a: float,
    S: int,
    g_min: float = 0.25,
):
    """
    Compute timestep to solve TV flow.

    `step_TV` is a forward Euler step of
        ∂u/∂t = Σ_i div_i (G_inv[i] A_i u / g),
    where the A_i are finite differences and g is the regularised gradient
    norm. Freezing g, the right hand side is a linear operator -L. At scale
    index s, the spatial central differences are multiplied by a**s, so that
    ‖A_1‖, ‖A_2‖ ≤ a**s / dxy, while ‖A_3‖ ≤ 1 / dθ.

    Since g vanishes in flat regions, there is no timestep that is stable for
    all images. Due to the nonlinearity, the scheme does not blow up even if
    the bound below is violated, but instead small oscillations appear in flat
    regions. We therefore require stability only where the gradient is at
    least `g_min` grey levels per pixel, as for SE(2). Because of the factor
    a**s, this corresponds to g ≥ a**s g_min at scale index s, so that
        ‖L‖ ≤ ((G_inv[0] + G_inv[1]) a**(S-1) / dxy**2
               + G_inv[2] / dθ**2 + G_inv[3] ‖L_4‖) / g_min,
    with L_4 the operator in the scale direction, and forward Euler is stable
    if dt ≤ 2 / ‖L‖. For S = 1, this reduces to the timestep for SE(2).

    The default `g_min` of 1/4 matches the standard explicit diffusion bound
    1 / (2 Σ_i G_inv[i] / h_i**2) for the finest scale.

    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_inv`: tuple[float, float, float, float] of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
        `a`: dilation factor between consecutive scales, taking values greater
          than 1.
        `S`: number of scales.
      Optional:
        `g_min`: lower bound of the regularised gradient norm, in grey levels
          per pixel, for which the scheme is guaranteed to be stable, taking
          values greater than 0. Defaults to 1/4.

    Returns:
        timestep, taking values greater than 0.
    """
    max_a = a**(S - 1)
    return 2 * g_min / (
        (G_inv[0] + G_inv[1]) * max_a / dxy**2
        + G_inv[2] / dθ**2
        + G_inv[3] * _scale_operator_norm(a, S)
    )
