"""Low-rank linear layers for SALAD-compressed and LoRA finetuning.

LowRankLinear:  W ≈ A·B  (full replacement, both A and B trainable)
LoRALinear:     y = Wx + (α/r)·BAx  (frozen base + trainable adapters)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.out_features, self.r = A.shape
        r_b, self.in_features = B.shape
        assert r_b == self.r

        self.A = nn.Parameter(A.contiguous())
        self.B = nn.Parameter(B.contiguous())
        if bias is not None:
            self.bias = nn.Parameter(bias.contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x @ self.B.T) @ self.A.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"rank={self.r}, bias={self.bias is not None}")


class LoRALinear(nn.Module):
    """Frozen base weight + trainable low-rank adapter: y = base(x) + (α/r)·B(Ax).

    Base can be nn.Linear or LowRankLinear — either way it's frozen as a unit.
    """

    def __init__(self, base: nn.Linear | LowRankLinear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def extra_repr(self) -> str:
        return (f"in_features={self.base.in_features}, "
                f"out_features={self.base.out_features}, "
                f"rank={self.r}, scaling={self.scaling}")
