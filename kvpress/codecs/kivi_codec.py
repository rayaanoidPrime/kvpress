# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kvpress.codecs.base_codec import KVCodec


def _pack(flat_u8: Tensor, bits: int) -> Tensor:
    if bits == 8:
        return flat_u8
    elif bits == 4:
        n = flat_u8.numel()
        if n % 2 != 0:
            flat_u8 = torch.cat([flat_u8, torch.zeros(1, dtype=torch.uint8)])
        lo = flat_u8[0::2] & 0x0F
        hi = (flat_u8[1::2] & 0x0F) << 4
        return lo | hi
    elif bits == 2:
        n = flat_u8.numel()
        rem = n % 4
        if rem != 0:
            flat_u8 = torch.cat([flat_u8, torch.zeros(4 - rem, dtype=torch.uint8)])
        b0 = flat_u8[0::4] & 0x3
        b1 = (flat_u8[1::4] & 0x3) << 2
        b2 = (flat_u8[2::4] & 0x3) << 4
        b3 = (flat_u8[3::4] & 0x3) << 6
        return b0 | b1 | b2 | b3
    else:
        raise ValueError(f"Unsupported bits: {bits}")


@dataclass
class KIVICodec(KVCodec):
    bits: int = 4
    group_size: int = 64
    mode: str = "auto"

    def __post_init__(self):
        assert self.bits in (2, 4, 8), "bits must be 2, 4, or 8"
        assert self.group_size > 0
        assert self.mode in ("auto", "key", "value")

    def encode(self, x: Tensor) -> tuple[Any, dict]:
        is_key = self.mode != "value"
        return self._quantize(x, is_key)

    def decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        x_q, _packed = compressed
        B, H, S, D = original_shape
        is_key = meta["is_key"]
        pad = meta["pad"]
        x_q_shape = meta["x_q_shape"]
        n_groups_dim = meta["n_groups_dim"]
        gs = meta["group_size"]

        g_min_flat = torch.tensor(meta["g_min"], dtype=torch.float16).float()
        scale_flat = torch.tensor(meta["scale"], dtype=torch.float16).float()

        x_q = x_q.reshape(x_q_shape)

        if is_key:
            g_min_r = g_min_flat.reshape(B, H, n_groups_dim, 1, D)
            scale_r = scale_flat.reshape(B, H, n_groups_dim, 1, D)
            x_hat_g = x_q.float() * scale_r + g_min_r
            x_hat = x_hat_g.reshape(B, H, n_groups_dim * gs, D)
            if pad > 0:
                x_hat = x_hat[:, :, :S, :]
        else:
            g_min_r = g_min_flat.reshape(B, H, S, n_groups_dim, 1)
            scale_r = scale_flat.reshape(B, H, S, n_groups_dim, 1)
            x_hat_g = x_q.float() * scale_r + g_min_r
            x_hat = x_hat_g.reshape(B, H, S, n_groups_dim * gs)
            if pad > 0:
                x_hat = x_hat[:, :, :, :D]

        original_dtype_str = meta["original_dtype"]
        if original_dtype_str == "float16":
            x_hat = x_hat.half()
        elif original_dtype_str == "bfloat16":
            x_hat = x_hat.bfloat16()
        return x_hat

    def _compressed_bytes(self, compressed: Any, meta: dict) -> int:
        _x_q, packed = compressed
        n_groups_scalar = int(meta["n_groups_scalar"])
        data_bytes = packed.numel()
        scale_bytes = n_groups_scalar * 2
        zero_bytes = n_groups_scalar * 2
        return data_bytes + scale_bytes + zero_bytes

    def _quantize(self, x: Tensor, is_key: bool) -> tuple[Any, dict]:
        x = x.float()
        B, H, S, D = x.shape
        n_levels = (1 << self.bits) - 1

        if is_key:
            gs = min(self.group_size, S)
            n_groups_dim = int(math.ceil(S / gs))
            pad = n_groups_dim * gs - S
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            x_g = x.reshape(B, H, n_groups_dim, gs, D)
            g_min = x_g.amin(dim=-2, keepdim=True)
            g_max = x_g.amax(dim=-2, keepdim=True)
        else:
            gs = min(self.group_size, D)
            n_groups_dim = int(math.ceil(D / gs))
            pad = n_groups_dim * gs - D
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, pad))
            x_g = x.reshape(B, H, S, n_groups_dim, gs)
            g_min = x_g.amin(dim=-1, keepdim=True)
            g_max = x_g.amax(dim=-1, keepdim=True)

        scale = (g_max - g_min).clamp(min=1e-8) / n_levels
        x_q = ((x_g - g_min) / scale).round().clamp(0, n_levels).to(torch.uint8)

        packed = _pack(x_q.flatten(), self.bits)
        n_groups_scalar = int(g_min.numel())

        meta = {
            "bits": self.bits,
            "group_size": gs,
            "n_groups_dim": n_groups_dim,
            "n_groups_scalar": n_groups_scalar,
            "is_key": is_key,
            "pad": pad,
            "original_shape": tuple(x.shape),
            "original_dtype": str(x.dtype).split(".")[-1],
            "g_min": g_min.flatten().half().tolist(),
            "scale": scale.flatten().half().tolist(),
            "x_q_shape": tuple(x_q.shape),
        }
        return (x_q.flatten(), packed), meta
