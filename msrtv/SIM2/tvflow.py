"""
tvflow
======

Compute the SIM(2) TV Flow, using PyTorch.[1] The primary methods are:
  1. `denoise`: perform multi-scale roto-translational TV flow on SIM(2) for
  denoising,[1] extending roto-translational total variation (RT-TV) flow on
  SE(2).[2][3]
  2. `compute_timestep`: compute a timestep for which the explicit scheme is
  stable.

References:
  [1]: "Multi-Scale Roto-Translational Total Variation Flow."
  [2]: A. Chambolle and Th. Pock.
    "Total roto-translational variation." In: Numerische Mathematik 142
    (2019), pp. 611--666.
    DOI:10.1007/s00211-019-01026-w.
  [3]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
    "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
    Positions and Orientations." In: Journal of Mathematical Imaging and
    Vision 63 (2021), pp. 237--262.
    DOI:10.1007/s10851-020-00991-4.
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
    Perform multi-scale roto-translational TV flow enhancing in SIM(2),[1]
    extending roto-translational total variation (RT-TV) flow[2] using an
    adaptation of the algorithm described by Smets et al.[3]

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
        [1]: "Multi-Scale Roto-Translational Total Variation Flow."
        [2]: A. Chambolle and Th. Pock.
          "Total roto-translational variation." In: Numerische Mathematik 142
          (2019), pp. 611--666.
          DOI:10.1007/s00211-019-01026-w.
        [3]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
          "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
          Positions and Orientations." In: Journal of Mathematical Imaging and
          Vision 63 (2021), pp. 237--262.
          DOI:10.1007/s10851-020-00991-4.
    """
    S = u.shape[-4]
    Or = u.shape[-3]
    dθ = 2 * pi / Or
    dt_opt = compute_timestep(1.0, dθ, G_inv, a, S)
    n = ceil(T / dt_opt)
    dt = T / n

    if ground_truth is not None:
        PSNRs = [
            PSNR(
                inverse_wavelet_transform(u, ground_truth.u_low, ground_truth.filters, ground_truth.lowpass),
                ground_truth.ground_truth,
                max_val=max_val
            )
        ]
    else:
        PSNRs = []
        
    forward_derivatives = True
    for _ in tqdm(range(n)):
        u = step_TV(u, dt, G_inv, a, σ_s, σ_o, σ_a, forward_derivatives)
        forward_derivatives = not forward_derivatives
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
    forward_derivatives: bool,
) -> torch.Tensor:
    S, Or, H, W = u.shape
    k_A1_grad, k_A2_grad = get_A_kernels(Or, u.device, forward_derivatives)
    k_A1_div, k_A2_div = get_A_kernels(Or, u.device, not forward_derivatives)
    dθ = 2 * torch.pi / Or
    scales = (a ** torch.arange(S, device=u.device))

    u_reshaped = u.reshape(1, S * Or, H, W)
    u_pad = F.pad(u_reshaped, (1, 1, 1, 1), mode="replicate")
    A1_u = F.conv2d(u_pad, k_A1_grad.repeat(S, 1, 1, 1), groups=S * Or)
    A2_u = F.conv2d(u_pad, k_A2_grad.repeat(S, 1, 1, 1), groups=S * Or)

    A1_u = scales.view(S, 1, 1, 1) * A1_u.reshape(S, Or, H, W)
    A2_u = scales.view(S, 1, 1, 1) * A2_u.reshape(S, Or, H, W)

    u_padded = F.pad(
        u[None],
        pad=(0, 0, 0, 0, 1, 1),
        mode="circular",
    )[0]
    if forward_derivatives:
        A3_u = (u_padded[:, 2:] - u_padded[:, 1:-1]) / dθ
    else:
        A3_u = (u_padded[:, 1:-1] - u_padded[:, :-2]) / dθ

    u_padded = torch.cat((u[:1], u, u[-1:]), dim=0)
    dscale_forward = a - 1.
    dscale_backward = 1. - 1. / a
    if forward_derivatives:
        A4_u = (u_padded[2:] - u_padded[1:-1]) / dscale_forward
    else:
        A4_u = (u_padded[1:-1] - u_padded[:-2]) / dscale_backward

    grad_norm = (G_inv[0] * A1_u**2 + G_inv[1] * A2_u**2 + G_inv[2] * A3_u**2 + G_inv[3] * A4_u**2).sqrt()
    grad_norm = gaussian_blur_4d_separable(grad_norm, (σ_s, σ_s, σ_o, σ_a))  # + 1e-9

    normalised_A1_u = G_inv[0] * A1_u / grad_norm
    normalised_A2_u = G_inv[1] * A2_u / grad_norm
    normalised_A3_u = G_inv[2] * A3_u / grad_norm
    normalised_A4_u = G_inv[3] * A4_u / grad_norm

    normalised_A1_u_pad = F.pad(normalised_A1_u.reshape(1, S * Or, H, W), (1, 1, 1, 1), mode="constant")
    divergence_normalised_grad_1 =  F.conv2d(normalised_A1_u_pad, k_A1_div.repeat(S, 1, 1, 1), groups=S * Or)
    divergence_normalised_grad_1 = scales.view(S,1,1,1) * divergence_normalised_grad_1.reshape(S, Or, H, W)

    normalised_A2_u_pad = F.pad(normalised_A2_u.reshape(1, S * Or, H, W), (1, 1, 1, 1), mode="constant")
    divergence_normalised_grad_2 = F.conv2d(normalised_A2_u_pad, k_A2_div.repeat(S, 1, 1, 1), groups=S * Or)
    divergence_normalised_grad_2 = scales.view(S,1,1,1) * divergence_normalised_grad_2.reshape(S, Or, H, W)

    normalised_A3_u_padded = F.pad(
        normalised_A3_u[None],
        pad=(0, 0, 0, 0, 1, 1),
        mode="circular",
    )[0]
    if not forward_derivatives:
        divergence_normalised_grad_3 = (normalised_A3_u_padded[:, 2:] - normalised_A3_u_padded[:, 1:-1]) / dθ
    else:
        divergence_normalised_grad_3 = (normalised_A3_u_padded[:, 1:-1] - normalised_A3_u_padded[:, :-2]) / dθ

    normalised_A4_u_padded = F.pad(normalised_A4_u, (0, 0, 0, 0, 0, 0, 1, 1))
    if forward_derivatives:
        divergence_normalised_grad_4 = (normalised_A4_u_padded[1:-1] - a**2 * normalised_A4_u_padded[:-2]) / dscale_forward
    else:
        divergence_normalised_grad_4 = (normalised_A4_u_padded[2:] / a**2 - normalised_A4_u_padded[1:-1]) / dscale_backward

    return u + dt * (
        divergence_normalised_grad_1
        + divergence_normalised_grad_2
        + divergence_normalised_grad_3
        + divergence_normalised_grad_4
        
    )


def compute_timestep(
    dxy: float,
    dθ: float,
    G_inv: tuple[float, float, float, float],
    a: float,
    S: int,
):
    """
    Compute timestep to solve multi-scale roto-translational TV flow.[1]

    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_inv`: tuple[float, float, float, float] of constants of the inverse
          of the diagonal metric tensor with respect to left invariant basis.
        `a`: dilation factor between consecutive scales, taking values greater
          than 1.
        `S`: number of scales.

    Returns:
        timestep, taking values greater than 0.

    References:
        [1]: "Multi-Scale Roto-Translational Total Variation Flow."
          In: Scale Space and Variational Methods in Computer Vision (2027).
    """
    max_a = a**(S - 1)
    return 2 / (
        (G_inv[0] + G_inv[1]) * max_a / dxy**2
        + G_inv[2] / dθ**2
        + G_inv[3] * _scale_operator_norm(a, S)
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
