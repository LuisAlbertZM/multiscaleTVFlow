
import torch
import torch.nn.functional as F
from math import ceil, log10
from functools import lru_cache

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

def gaussian_blur_3d_separable(x, sigmas, truncate=4.0):
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


@lru_cache
def gaussian_kernel_1d(kernel_size, sigma, device=None, dtype=torch.float32):
    coords = (
        torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    )
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

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