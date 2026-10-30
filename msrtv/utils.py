
import torch
import torch.nn.functional as F
from math import ceil, log10
from functools import lru_cache


def gaussian_blur_4d_separable(
    x: torch.Tensor,
    sigmas: tuple[float, float, float, float],
    truncate: float = 4.0
) -> torch.Tensor:
    kernel_size = [1 + 2 * ceil(truncate * sigma) for sigma in sigmas]

    x = x[None]

    kx = gaussian_kernel_1d(kernel_size[0], sigmas[0], x.device, x.dtype).view(1, 1, 1, 1, -1)
    kx = kx.repeat(x.shape[1], 1, 1, 1, 1)

    ky = gaussian_kernel_1d(kernel_size[1], sigmas[1], x.device, x.dtype).view(1, 1, 1, -1, 1)
    ky = ky.repeat(x.shape[1], 1, 1, 1, 1)

    x = F.pad(x, (kernel_size[0] // 2, kernel_size[0] // 2, 0, 0, 0, 0), mode="reflect")
    x = F.conv3d(x, kx, groups=x.shape[1])

    x = F.pad(x, (0, 0, kernel_size[1] // 2, kernel_size[1] // 2, 0, 0), mode="reflect")
    x = F.conv3d(x, ky, groups=x.shape[1])

    x = x[0]

    kθ = gaussian_blur_matrix_circular(x.shape[1], kernel_size[2], sigmas[2], x.device, x.dtype)
    ka = gaussian_blur_matrix_reflect(x.shape[0], kernel_size[3], sigmas[3], x.device, x.dtype)
    x = torch.einsum("ts,ro,so...->tr...", ka, kθ, x)

    return x


def gaussian_blur_3d_separable(
    x: torch.Tensor,
    sigmas: tuple[float, float, float],
    truncate: float = 4.0
) -> torch.Tensor:
    kernel_size = [1 + 2 * ceil(truncate * sigma) for sigma in sigmas]

    x = x[None, None]

    kx = gaussian_kernel_1d(kernel_size[0], sigmas[0], x.device, x.dtype)
    ky = gaussian_kernel_1d(kernel_size[1], sigmas[1], x.device, x.dtype)
    kz = gaussian_kernel_1d(kernel_size[2], sigmas[2], x.device, x.dtype)

    kx = kx.view(1, 1, 1, 1, -1)
    ky = ky.view(1, 1, 1, -1, 1)
    kz = kz.view(1, 1, -1, 1, 1)

    x = F.pad(x, (kernel_size[0] // 2, kernel_size[0] // 2, 0, 0, 0, 0), mode="reflect")
    x = F.conv3d(x, kx)
    x = F.pad(x, (0, 0, kernel_size[1] // 2, kernel_size[1] // 2, 0, 0), mode="reflect")
    x = F.conv3d(x, ky)
    x = F.pad(
        x, (0, 0, 0, 0, kernel_size[2] // 2, kernel_size[2] // 2), mode="circular"
    )
    x = F.conv3d(x, kz)

    return x[0, 0]


def gaussian_blur_2d_separable(
    x: torch.Tensor,
    sigmas: tuple[float, float],
    truncate: float = 4.0
) -> torch.Tensor:
    kernel_size = [1 + 2 * ceil(truncate * sigma) for sigma in sigmas]

    x = x[None, None]

    kx = gaussian_kernel_1d(kernel_size[0], sigmas[0], x.device, x.dtype)
    ky = gaussian_kernel_1d(kernel_size[1], sigmas[1], x.device, x.dtype)

    kx = kx.view(1, 1, 1, -1)
    ky = ky.view(1, 1, -1, 1)

    x = F.pad(x, (kernel_size[0] // 2, kernel_size[0] // 2, 0, 0), mode="reflect")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, kernel_size[1] // 2, kernel_size[1] // 2), mode="reflect")
    x = F.conv2d(x, ky)

    return x[0, 0]


@lru_cache
def gaussian_kernel_1d(
    kernel_size: int,
    sigma: float,
    device: torch.DeviceObjType | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = (
        torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    )
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


@lru_cache
def gaussian_blur_matrix_reflect(
    n: int,
    kernel_size: int,
    sigma: float,
    device: torch.DeviceObjType | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Matrix of 1D Gaussian blurring of a signal of length `n` with reflecting
    boundary conditions, i.e. as `F.pad(mode="reflect")` followed by
    `F.conv1d`. Unlike `F.pad`, it allows kernels wider than the signal.
    """
    kernel = gaussian_kernel_1d(kernel_size, sigma, device, dtype)
    offsets = torch.arange(kernel_size, device=device) - kernel_size // 2
    indices = torch.arange(n, device=device)[:, None] + offsets[None]
    if n > 1:
        period = 2 * (n - 1)
        indices = indices % period
        indices = torch.where(indices >= n, period - indices, indices)
    else:
        indices = torch.zeros_like(indices)
    matrix = torch.zeros(n, n, device=device, dtype=dtype)
    matrix.scatter_add_(1, indices, kernel.expand(n, -1).contiguous())
    return matrix


@lru_cache
def gaussian_blur_matrix_circular(
    n: int,
    kernel_size: int,
    sigma: float,
    device: torch.DeviceObjType | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Matrix of 1D Gaussian blurring of a signal of length `n` with periodic
    boundary conditions, i.e. as `F.pad(mode="circular")` followed by
    `F.conv1d`. Unlike `F.pad`, it allows kernels wider than the signal.
    """
    kernel = gaussian_kernel_1d(kernel_size, sigma, device, dtype)
    offsets = torch.arange(kernel_size, device=device) - kernel_size // 2
    indices = (torch.arange(n, device=device)[:, None] + offsets[None]) % n
    matrix = torch.zeros(n, n, device=device, dtype=dtype)
    matrix.scatter_add_(1, indices, kernel.expand(n, -1).contiguous())
    return matrix


@lru_cache
def get_A_kernels(
    Or: int,
    device: str | torch.DeviceObjType,
    forward_derivatives: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    if forward_derivatives:
        return bilinear_shift_kernel(cos, sin), bilinear_shift_kernel(-sin, cos)
    else:
        return -bilinear_shift_kernel(-cos, -sin), -bilinear_shift_kernel(sin, -cos)


def PSNR(
    denoised: torch.Tensor,
    ground_truth: torch.Tensor,
    max_val: float = 255.,
) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) of `denoised` with respect to
    `ground_truth`.
    """
    mean_squared_difference = ((denoised - ground_truth)**2).mean()
    
    return 10 * log10(max_val**2 / mean_squared_difference)