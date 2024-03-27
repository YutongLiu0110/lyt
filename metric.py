import torch


def calc_nd(true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return (true - pred).abs().sum() / (true.abs().sum() + 1e-7)
