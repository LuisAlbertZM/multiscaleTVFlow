"""
threshold
=========

Perform coefficient thresholding on a continuous wavelet transform.
"""

import torch
import torch.nn.functional as F
from msrtv.utils import PSNR
from msrtv.SIM2.simoncellitransform import inverse_wavelet_transform
from msrtv.SIM2.utils import GroundTruth


def denoise(
    u: torch.Tensor,
    quantiles: torch.Tensor,
    ground_truth: GroundTruth | None = None,
    max_val: float = 255.
) -> tuple[torch.Tensor, list[float]]:
    thresholds = torch.quantile(
        u.abs().flatten(start_dim=-2),
        q=quantiles,
        dim=-1,
        keepdim=True,
    ).unsqueeze(-1)
    denoised = F.relu(u - thresholds) - F.relu(-u - thresholds)

    PSNRs = []
    if ground_truth is not None:
        for thresholded in denoised:
            PSNRs.append(PSNR(
                inverse_wavelet_transform(u, ground_truth.u_low, ground_truth.filters, ground_truth.lowpass),
                ground_truth.ground_truth,
                max_val=max_val
            ))

    return denoised, PSNRs
