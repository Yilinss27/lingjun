from __future__ import annotations

import torch


class TriangularCausalMask:
    def __init__(self, B: int, L: int, device: torch.device | str = "cpu"):
        mask_shape = (B, 1, L, L)
        self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class ProbMask:
    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: torch.Tensor,
        scores: torch.Tensor,
        device: torch.device | str = "cpu",
    ):
        base = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        expanded = base[None, None, :, :].expand(B, H, L, scores.shape[-1])
        indicator = expanded[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :,
        ]
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask
