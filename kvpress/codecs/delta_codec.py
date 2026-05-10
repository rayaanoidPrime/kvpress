# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from kvpress.codecs.base_codec import KVCodec


def _pack_uint4(flat_u8: Tensor) -> Tensor:
    n = flat_u8.numel()
    if n % 2 != 0:
        flat_u8 = torch.cat([flat_u8, torch.zeros(1, dtype=torch.uint8)])
    lo = flat_u8[0::2] & 0x0F
    hi = (flat_u8[1::2] & 0x0F) << 4
    return lo | hi


def _unpack_uint4(packed: Tensor, n_original: int) -> Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = torch.stack([lo, hi], dim=-1).flatten()
    return interleaved[:n_original]


@dataclass
class DeltaCodec(KVCodec):
    quantize_bits: Optional[int] = None
    per_head: bool = True

    def __post_init__(self):
        assert self.quantize_bits in (None, 4, 8), "quantize_bits must be None, 4, or 8"

    def encode(self, x: Tensor) -> tuple[Any, dict]:
        x = x.float()
        B, H, S, D = x.shape
        anchor = x[:, :, :1, :].half()

        if S == 1:
            meta = {
                "quantize_bits": None,
                "per_head": self.per_head,
                "original_shape": tuple(x.shape),
                "original_dtype": str(x.dtype).split(".")[-1],
                "delta_shape": (B, H, 0, D),
            }
            return (anchor, torch.empty((B, H, 0, D), dtype=torch.float16)), meta

        deltas = x[:, :, 1:, :] - x[:, :, :-1, :]

        if self.quantize_bits is None:
            deltas_stored = deltas.half()
            meta = {
                "quantize_bits": None,
                "per_head": self.per_head,
                "original_shape": tuple(x.shape),
                "original_dtype": str(x.dtype).split(".")[-1],
                "delta_shape": tuple(deltas.shape),
            }
            return (anchor, deltas_stored), meta

        n_levels = (1 << self.quantize_bits) - 1

        if self.per_head:
            d_min = deltas.amin(dim=(-2, -1), keepdim=True)
            d_max = deltas.amax(dim=(-2, -1), keepdim=True)
        else:
            d_min_val = float(deltas.min())
            d_max_val = float(deltas.max())
            d_min = torch.tensor(d_min_val).reshape(1, 1, 1, 1).expand(B, H, 1, 1)
            d_max = torch.tensor(d_max_val).reshape(1, 1, 1, 1).expand(B, H, 1, 1)

        d_range = (d_max - d_min).clamp(min=1e-8)
        scale = d_range / n_levels
        delta_q = ((deltas - d_min) / scale).round().clamp(0, n_levels).to(torch.uint8)

        if self.quantize_bits == 4:
            packed = _pack_uint4(delta_q.flatten())
        else:
            packed = delta_q

        meta = {
            "quantize_bits": self.quantize_bits,
            "per_head": self.per_head,
            "d_min": d_min.squeeze().tolist(),
            "scale": scale.squeeze().tolist(),
            "original_shape": tuple(x.shape),
            "original_dtype": str(x.dtype).split(".")[-1],
            "delta_shape": tuple(deltas.shape),
        }
        return (anchor, packed), meta

    def decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        anchor, delta_data = compressed
        anchor = anchor.float()
        B, H, S, D = original_shape
        quantize_bits = meta["quantize_bits"]

        if S == 1:
            return anchor.reshape(original_shape)

        delta_shape = meta["delta_shape"]

        if quantize_bits is None:
            deltas = delta_data.float()
        else:
            if quantize_bits == 4:
                n_delta = 1
                for s in delta_shape:
                    n_delta *= s
                delta_q_u8 = _unpack_uint4(delta_data, n_delta).reshape(delta_shape)
            else:
                delta_q_u8 = delta_data

            d_min_tensor = torch.tensor(meta["d_min"], dtype=torch.float32)
            scale_tensor = torch.tensor(meta["scale"], dtype=torch.float32)
            per_head_flag = meta["per_head"]

            if per_head_flag:
                d_min_tensor = d_min_tensor.reshape(B, H, 1, 1)
                scale_tensor = scale_tensor.reshape(B, H, 1, 1)
            else:
                d_min_tensor = d_min_tensor.reshape(1, 1, 1, 1)
                scale_tensor = scale_tensor.reshape(1, 1, 1, 1)

            deltas = delta_q_u8.float() * scale_tensor + d_min_tensor

        zero = torch.zeros_like(anchor)
        all_d = torch.cat([zero, deltas], dim=2)
        recon = anchor + all_d.cumsum(dim=2)

        original_dtype_str = meta["original_dtype"]
        if original_dtype_str == "float16":
            recon = recon.half()
        elif original_dtype_str == "bfloat16":
            recon = recon.bfloat16()
        return recon

    def _compressed_bytes(self, compressed: Any, meta: dict) -> int:
        anchor, delta_data = compressed
        anchor_bytes = anchor.numel() * 2
        quantize_bits = meta["quantize_bits"]

        if quantize_bits is None:
            delta_bytes = delta_data.numel() * 2
        elif quantize_bits == 4:
            delta_bytes = delta_data.numel()
        else:
            delta_bytes = delta_data.numel()

        metadata_bytes = 32
        return anchor_bytes + delta_bytes + metadata_bytes
