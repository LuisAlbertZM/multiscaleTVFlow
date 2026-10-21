import torch
from dataclasses import dataclass


@dataclass
class GroundTruth:
    ground_truth: torch.Tensor
    u_low: torch.Tensor
    filters: torch.Tensor
    lowpass: torch.Tensor