"""
SIM(2)TVFlowPyTorch
=====================

Compute the SIM(2) TV Flow, using PyTorch.
"""

import torch
import torch.nn.functional as F
from math import ceil, pi
from tqdm.auto import tqdm
from functools import lru_cache


def TV_enhancing(
    u: torch.Tensor,
    T: float,
    G_inv: tuple[float, float, float, float],
    a: float,
    σ_s: float,
    σ_o: float,
    σ_a: float,
) -> torch.Tensor:
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
    dθ = 2 * pi / (Or - 1)
    da = torch.log(torch.tensor(a, device=u.device, dtype=u.dtype))
    max_a = a**(S-1)
    dt_opt = compute_timestep(1.0, dθ, da, G_inv, max_a)
    n = ceil(T / dt_opt)
    dt = T / n

    for _ in tqdm(range(n)):
        u = step_TV(u, dt, G_inv, a, σ_s, σ_o, σ_a)
    return u


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
    dθ = 2 * torch.pi / (Or - 1)
    scales = (a ** torch.arange(S, device=u.device))
    da = torch.log(torch.tensor(a, device=u.device))

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
    #print(A1_u.max(), A2_u.max(), A3_u.max())

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

def compute_timestep(dxy, dθ, da, G_inv, max_a):
    """
    Compute timestep to solve TV flow.

    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `ds`: step size in a direction, taking values greater than 0.
        `G_inv`: tuple[float, float, float, float] of constants of the inverse of the
          diagonal metric tensor with respect to left invariant basis.

    Returns:
        timestep, taking values greater than 0.
    """
    # dt = 1.0 / (
    #     2.0 * (
    #         G_inv[0] * max_a**2 / dxy**2 +
    #         G_inv[1] * max_a**2 / dxy**2 +
    #         G_inv[2] / dθ**2 +
    #         G_inv[3] / da**2
    #     )
    # )
    return dxy**2 * dθ * da / (2 * (
        G_inv[0] * dxy +
        G_inv[1] * dxy +
        G_inv[2] * dθ +
        G_inv[3] * da
    ))




@lru_cache
def get_A_kernels(Or: int, device: str | torch.DeviceObjType):
    θs = torch.arange(Or, device=device) * 2 * torch.pi / Or

    cos = torch.cos(θs)
    sin = torch.sin(θs)

    def bilinear_shift_kernel(dx, dy):
        k = torch.zeros(Or, 1, 3, 3, device=device)

        x_neg = (-dx).clamp(min=0)
        x_pos = dx.clamp(min=0)

        y_neg = (-dy).clamp(min=0)
        y_pos = dy.clamp(min=0)

        x_ctr = 1.0 - x_neg - x_pos
        y_ctr = 1.0 - y_neg - y_pos

        wx = torch.stack([x_neg, x_ctr, x_pos], dim=-1)
        wy = torch.stack([y_neg, y_ctr, y_pos], dim=-1)

        k[:, 0] = wy[:, :, None] * wx[:, None, :]
        k[..., 1, 1] += -1.0

        return k

    A1_forward = bilinear_shift_kernel(cos, sin)
    A1_backward = bilinear_shift_kernel(-cos, -sin)
    k_A1 = (A1_forward - A1_backward) / 2.0

    A2_forward = bilinear_shift_kernel(-sin, cos)
    A2_backward = bilinear_shift_kernel(sin, -cos)
    k_A2 = (A2_forward - A2_backward) / 2.0

    return k_A1, k_A2


def gaussian_blur_4d_separable(x, sigmas, truncate=4.0):
    kernel_size = [1 + 2 * ceil(truncate * sigma) for sigma in sigmas]

    x = x[None]

    kx = gaussian_kernel_1d(kernel_size[0], sigmas[0], x.device, x.dtype).view(1, 1, 1, 1, -1)
    kx = kx.repeat(x.shape[1], 1, 1, 1, 1)

    ky = gaussian_kernel_1d(kernel_size[1], sigmas[1], x.device, x.dtype).view(1, 1, 1, -1, 1)
    ky = ky.repeat(x.shape[1], 1, 1, 1, 1)

    kθ = gaussian_kernel_1d(kernel_size[2], sigmas[2], x.device, x.dtype).view(1, 1, -1, 1, 1)
    kθ = kθ.repeat(x.shape[1], 1, 1, 1, 1)
   
    x = F.pad(x, (kernel_size[0] // 2, kernel_size[0] // 2, 0, 0, 0, 0), mode="reflect")
    x = F.conv3d(x, kx, groups=x.shape[1])

    x = F.pad(x, (0, 0, kernel_size[1] // 2, kernel_size[1] // 2, 0, 0), mode="reflect")
    x = F.conv3d(x, ky, groups=x.shape[1])

    x = F.pad(x, (0, 0, 0, 0, kernel_size[2] // 2, kernel_size[2] // 2), mode="circular")
    x = F.conv3d(x, kθ, groups=x.shape[1])

    x = x[0]

    ka = gaussian_kernel_1d(kernel_size[3], sigmas[3], x.device, x.dtype).view(1, 1, -1)

    x_scale = (x.permute(1, 2, 3, 0).reshape(-1, 1, x.shape[0]))
    x_scale = F.pad(x_scale, (kernel_size[3] // 2, kernel_size[3] // 2), mode="reflect")

    x_scale = F.conv1d(x_scale, ka)

    x = (
        x_scale.reshape(
            x.shape[1],
            x.shape[2],
            x.shape[3],
            x.shape[0],
        )
        .permute(3, 0, 1, 2)
    )

    return x


@lru_cache
def gaussian_kernel_1d(kernel_size, sigma, device=None, dtype=torch.float32):
    coords = (
        torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    )
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel
