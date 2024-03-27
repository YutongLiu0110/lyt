import math
from typing import Tuple, Union

import torch
from torch import nn

from .Layers import ConvLayer, MLPLayer


class Gene(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            pred_len: int,
            attn_module: nn.Module,
            hidden_dim: Union[int, Tuple[int, int]],
            kernel_size: Union[int, Tuple[int, int]],
    ) -> None:
        super().__init__()
        self.pred_len = pred_len

        self.attn = attn_module
        self.enc = Encoder(feat_dim, hidden_dim, kernel_size)
        self.dec = Decoder(feat_dim, hidden_dim)

    def forward(
            self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for _ in range(self.pred_len):
            pattern, value = self.enc(data)
            rep, (query, key) = self.attn(pattern, value)
            pred = self.dec(rep)
            data = torch.cat([data, pred[..., -1:]], dim=-1)
        return pred, (query, key, value)

    # predict 方法通过调用 forward 方法获取预测结果，并从预测结果中取出最后 t 个时间步的数据返回。
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        return self.forward(data)[..., -self.pred_len:]


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

    attn_logits = torch.exp(
        query.transpose(0, 1) @ key
        - (query.transpose(0, 1) @ key)
        * torch.eye(query.shape[-1], device=query.device)
        / math.sqrt(query.shape[-1])
    )
    attn_scores = torch.softmax(attn_logits, dim=-1)
    attn_values = (attn_scores @ value.transpose(0, 1)).transpose(0, 1)
    return attn_values


# ShareAttn实现了共享注意力机制，通过对输入的模式嵌入和值进行处理，计算注意力加权后的输出，并提供查询和键作为辅助输出。
class ShareAttn(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.mlp_q = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim,
        )
        self.mlp_k = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )
        self.mlp_o = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )

    def forward(
            self, pattern: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query, key = self.mlp_q(pattern), self.mlp_k(pattern)
        # interpolation mode
        rep_in = self.mlp_o(calc_attn(query, key, value))
        # extrapolation mode
        rep_ex = self.mlp_o(
            calc_attn(query, key, value, kernel_size=self.kernel_size)
        )
        return torch.cat([rep_in, rep_ex], dim=-1), (query, key)


class Encoder(nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: Union[int, Tuple[int, int]],
            kernel_size: Union[int, Tuple[int, int]],
    ) -> None:
        super().__init__()
        self.mlp_v = MLPLayer(
            input_dim=feat_dim,
            # output_dim=feat_dim - 1 if feat_dim > 1 else feat_dim,
            output_dim=feat_dim if feat_dim > 1 else feat_dim,
            hidden_dim=hidden_dim,
        )
        self.conv_p = ConvLayer(
            input_dim=feat_dim,
            # output_dim=feat_dim - 1 if feat_dim > 1 else feat_dim,
            output_dim=feat_dim if feat_dim > 1 else feat_dim,
            hidden_dim=hidden_dim[0] if isinstance(hidden_dim, tuple) else hidden_dim,
            kernel_size=kernel_size,
        )

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.conv_p(data), self.mlp_v(data)
        # 得到源域和目标域的V矩阵


class Decoder(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp_z = MLPLayer(
            input_dim=feat_dim, output_dim=feat_dim, hidden_dim=hidden_dim
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        return self.mlp_z(rep)
