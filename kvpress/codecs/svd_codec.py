# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kvpress.codecs.base_codec import KVCodec


@dataclass
class SVDCodec(KVCodec):
    rank_ratio: float = 0.5
    per_head: bool = True

    def __post_init__(self):
        assert 0 < self.rank_ratio <= 1, "rank_ratio must be in (0, 1]"

    def encode(self, x: Tensor) -> tuple[Any, dict]:
        x = x.float()
        B, H, S, D = x.shape
        k = max(1, int(D * self.rank_ratio))

        if self.per_head:
            x_rh = x.reshape(B * H, S, D)
            U_list, S_list, Vt_list = [], [], []
            for i in range(x_rh.shape[0]):
                Ui, Si, Vti = torch.linalg.svd(x_rh[i], full_matrices=False)
                U_list.append(Ui[:, :k])
                S_list.append(Si[:k])
                Vt_list.append(Vti[:k, :])
            U = torch.stack(U_list).reshape(B, H, S, k).half()
            S = torch.stack(S_list).reshape(B, H, k).half()
            Vt = torch.stack(Vt_list).reshape(B, H, k, D).half()
        else:
            x_flat = x.reshape(B * H * S, D)
            U_all, S_all, Vt_all = torch.linalg.svd(x_flat, full_matrices=False)
            Uk = U_all[:, :k].reshape(B, H, S, k).half()
            Sk = S_all[:k].reshape(1, 1, k).expand(B, H, k).half()
            Vtk = Vt_all[:k, :].reshape(1, 1, k, D).expand(B, H, k, D).half()
            U, S, Vt = Uk, Sk, Vtk

        meta = {
            "rank": k,
            "rank_ratio": self.rank_ratio,
            "per_head": self.per_head,
            "original_shape": tuple(x.shape),
            "original_dtype": str(x.dtype).split(".")[-1],
        }
        return (U, S, Vt), meta

    def decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        U, S_svd, Vt = compressed
        U = U.float()
        S_svd = S_svd.float()
        Vt = Vt.float()

        U_scaled = U * S_svd.unsqueeze(-2)
        x_hat = U_scaled @ Vt

        original_dtype_str = meta["original_dtype"]
        if original_dtype_str == "float16":
            x_hat = x_hat.half()
        elif original_dtype_str == "bfloat16":
            x_hat = x_hat.bfloat16()
        return x_hat

    def _compressed_bytes(self, compressed: Any, meta: dict) -> int:
        U, S_svd, Vt = compressed
        return (int(U.numel()) + int(S_svd.numel()) + int(Vt.numel())) * 2
