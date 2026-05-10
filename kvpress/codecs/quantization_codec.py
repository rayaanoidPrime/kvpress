# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kvpress.codecs.base_codec import KVCodec


@dataclass
class QuantizationCodec(KVCodec):
    bits: int = 8

    def __post_init__(self):
        assert self.bits in (4, 8), "bits must be 4 or 8"

    def encode(self, x: Tensor) -> tuple[Any, dict]:
        x = x.float()
        x_min = float(x.min())
        x_max = float(x.max())
        n_levels = (1 << self.bits) - 1
        scale = (x_max - x_min) / n_levels
        scale = max(scale, 1e-8)
        x_q = ((x - x_min) / scale).round().clamp(0, n_levels).to(torch.uint8)

        if self.bits == 4:
            flat = x_q.flatten()
            n = flat.numel()
            even_n = (n + 1) // 2 * 2
            if n % 2 != 0:
                flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
            lo = flat[0::2] & 0x0F
            hi = (flat[1::2] & 0x0F) << 4
            packed = lo | hi
            compressed = packed
        else:
            compressed = x_q

        meta = {
            "min": x_min,
            "scale": scale,
            "bits": self.bits,
            "original_shape": tuple(x.shape),
            "original_dtype": str(x.dtype).split(".")[-1],
        }
        return compressed, meta

    def decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        bits = meta["bits"]
        x_min = meta["min"]
        scale = meta["scale"]
        original_dtype_str = meta["original_dtype"]

        if bits == 4:
            packed = compressed
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            x_q_u8 = torch.stack([lo, hi], dim=-1).flatten()
            n_original = 1
            for s in original_shape:
                n_original *= s
            x_q_u8 = x_q_u8[:n_original].reshape(original_shape)
        else:
            x_q_u8 = compressed

        x_hat = x_q_u8.float() * scale + x_min
        x_hat = x_hat.reshape(original_shape)
        if original_dtype_str == "float16":
            x_hat = x_hat.half()
        elif original_dtype_str == "bfloat16":
            x_hat = x_hat.bfloat16()
        return x_hat

    def _compressed_bytes(self, compressed: Any, meta: dict) -> int:
        return int(compressed.numel()) + 8
