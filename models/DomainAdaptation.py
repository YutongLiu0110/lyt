import math
from typing import Tuple, Union

import torch
from torch import nn

from .Layers import MLPLayer


def calc_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kernel_size: Union[int, Tuple[int, int]] = None,
) -> torch.Tensor:
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    if kernel_size:
        unit_len = (kernel_size - 1) // 2
        query = query[..., -1 - unit_len: -1 - unit_len + 1]
        key = key[..., kernel_size - unit_len - 1: -1 - unit_len - 1]
        value = value[..., kernel_size:-1]
    attn_logit = torch.exp(
        query.transpose(0, 1) @ key
        - (query.transpose(0, 1) @ key)
        * torch.eye(query.shape[-1], device=query.device)
        / math.sqrt(query.shape[-1])
    )
    attn_scores = torch.softmax(attn_logit, dim=-1)
    attn_values = (attn_scores @ value.transpose(0, 1)).transpose(0, 1)
    return attn_values


class DomainAdaptation(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.q = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim, is_cls=False
        )
        self.k = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim, is_cls=False
        )
        self.v = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim, is_cls=False
        )

    def forward(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_tgt_attn_value = calc_attn(query, key, value)
        dom_query = self.q(src_tgt_attn_value)
        dom_key = self.k(src_tgt_attn_value)
        return dom_query, dom_key
