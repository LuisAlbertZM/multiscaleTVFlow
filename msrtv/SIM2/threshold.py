"""
threshold
=========

Perform coefficient thresholding on a continuous wavelet transform. The
primary method is:
  1. `denoise`: denoise by soft-thresholding the wavelet coefficients.[1]

References:
  [1]: D.L. Donoho.
    "De-Noising by Soft-Thresholding." In: IEEE Transactions on
    Information Theory 41 (1995), pp. 613--627.
    DOI:10.1109/18.382009.
"""

import torch
from torch.nn.functional import relu
from msrtv.utils import PSNR
from msrtv.SIM2.simoncellitransform import inverse_wavelet_transform
from msrtv.SIM2.utils import GroundTruth
from tqdm.auto import tqdm


def denoise(
    u: torch.Tensor,
    quantiles: torch.Tensor,
    ground_truth: GroundTruth | None = None,
    max_val: float = 255.
) -> tuple[torch.Tensor, list[float]]:
    """
    Denoise by soft-thresholding the wavelet coefficients, as described by
    Donoho.[1] For each quantile `q`, the threshold of each orientation (and
    scale) is the `q`-quantile of the absolute values of its coefficients.

    Args:
        `u`: torch.Tensor wavelet transform of the noisy image, with shape
          [..., H, W].
        `quantiles`: torch.Tensor of quantiles used to determine the
          thresholds, taking values between 0 and 1.
        `ground_truth`: GroundTruth instance for computing PSNR.
          groundtruth: torch.Tensor ground truth, with shape [H, W].
          u_low: torch.Tensor low passed noisy image, with shape [H, W].
          filters: torch.Tensor filters used for lifting.
          lowpass: torch.Tensor lowpass filter used for lifting.
        `max_val`: maximum value images take, for computing PSNR.

    Returns:
        torch.Tensor soft-thresholded wavelet transform for the last quantile,
        and list of PSNRs for each quantile.

    References:
        [1]: D.L. Donoho.
          "De-Noising by Soft-Thresholding." In: IEEE Transactions on
          Information Theory 41 (1995), pp. 613--627.
          DOI:10.1109/18.382009.
    """
    PSNRs = []
    for q in tqdm(quantiles):
        threshold = torch.quantile(
            u.abs().flatten(start_dim=-2),
            q=q,
            dim=-1,
            keepdim=True,
        ).unsqueeze(-1)
        denoised = relu(u - threshold) - relu(-u - threshold)

        if ground_truth is not None:
            PSNRs.append(PSNR(
                inverse_wavelet_transform(denoised, ground_truth.u_low, ground_truth.filters, ground_truth.lowpass),
                ground_truth.ground_truth,
                max_val=max_val
            ))

    return denoised, PSNRs
