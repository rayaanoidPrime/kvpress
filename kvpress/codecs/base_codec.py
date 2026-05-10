# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class CodecStats:
    compression_ratio: float
    compressed_bytes: int
    original_bytes: int
    encode_ms: float
    decode_ms: float
    mse: float
    cosine_sim: float
    max_abs_err: float
    attn_logit_rel_err: float

    def to_dict(self) -> dict:
        return {
            "compression_ratio": self.compression_ratio,
            "compressed_bytes": self.compressed_bytes,
            "original_bytes": self.original_bytes,
            "encode_ms": self.encode_ms,
            "decode_ms": self.decode_ms,
            "mse": self.mse,
            "cosine_sim": self.cosine_sim,
            "max_abs_err": self.max_abs_err,
            "attn_logit_rel_err": self.attn_logit_rel_err,
        }


class KVCodec:
    def encode(self, x: Tensor) -> tuple[Any, dict]:
        raise NotImplementedError

    def decode(self, compressed: Any, meta: dict, original_shape: tuple) -> Tensor:
        raise NotImplementedError

    def _compressed_bytes(self, compressed: Any, meta: dict) -> int:
        raise NotImplementedError

    def roundtrip(
        self,
        x: Tensor,
        n_warmup: int = 3,
        n_encode_trials: int = 10,
        n_decode_trials: int = 50,
        simulated_query_heads: int = 8,
    ) -> CodecStats:
        x = x.float()

        compressed, meta = self.encode(x)

        for _ in range(n_warmup):
            _ = self.decode(compressed, meta, x.shape)

        encode_times = []
        for _ in range(n_encode_trials):
            t0 = time.perf_counter()
            _ = self.encode(x)
            encode_times.append(time.perf_counter() - t0)

        decode_times = []
        x_hat = None
        for _ in range(n_decode_trials):
            t0 = time.perf_counter()
            x_hat = self.decode(compressed, meta, x.shape)
            decode_times.append(time.perf_counter() - t0)

        encode_ms = sum(encode_times) / len(encode_times) * 1000.0
        decode_ms = sum(decode_times) / len(decode_times) * 1000.0

        compressed_bytes = self._compressed_bytes(compressed, meta)
        original_bytes = x.numel() * 2
        compression_ratio = 1.0 - compressed_bytes / max(original_bytes, 1)

        mse = float((x - x_hat).pow(2).mean())
        x_flat = x.reshape(-1)
        xh_flat = x_hat.reshape(-1)
        cos_num = float((x_flat * xh_flat).sum())
        cos_den = float((x_flat.norm() * xh_flat.norm()).clamp(min=1e-12))
        cosine_sim = cos_num / cos_den
        max_abs_err = float((x - x_hat).abs().max())

        B, H, S, D = x.shape
        if H < simulated_query_heads:
            n_repeat = simulated_query_heads // H
            k = x.repeat_interleave(n_repeat, dim=1)
            k_hat = x_hat.repeat_interleave(n_repeat, dim=1)
            n_q_heads = simulated_query_heads
        else:
            k = x
            k_hat = x_hat
            n_q_heads = H

        try:
            Q = torch.randn(B, n_q_heads, 1, D, dtype=torch.float32)
            logits = torch.matmul(Q, k.transpose(-2, -1)) / (D**0.5)
            logits_hat = torch.matmul(Q, k_hat.transpose(-2, -1)) / (D**0.5)
            num = (logits - logits_hat).norm()
            den = logits.norm().clamp(min=1e-12)
            attn_logit_rel_err = float(num / den)
        except Exception:
            attn_logit_rel_err = float("nan")

        return CodecStats(
            compression_ratio=compression_ratio,
            compressed_bytes=compressed_bytes,
            original_bytes=original_bytes,
            encode_ms=encode_ms,
            decode_ms=decode_ms,
            mse=mse,
            cosine_sim=cosine_sim,
            max_abs_err=max_abs_err,
            attn_logit_rel_err=attn_logit_rel_err,
        )
