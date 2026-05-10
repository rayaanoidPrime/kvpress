# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.codecs.base_codec import KVCodec
from kvpress.codecs.kivi_codec import KIVICodec
from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class CodecPress(BasePress):
    key_codec: KVCodec = None
    value_codec: KVCodec = None
    compression_ratio: float = 0.0

    layer_encode_ms: dict = field(default_factory=dict, init=False)
    layer_decode_ms: dict = field(default_factory=dict, init=False)
    layer_mse: dict = field(default_factory=dict, init=False)
    layer_cosine_sim: dict = field(default_factory=dict, init=False)
    layer_compressed_bytes: dict = field(default_factory=dict, init=False)
    layer_original_bytes: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        assert self.key_codec is not None, "key_codec must be set"
        assert self.value_codec is not None, "value_codec must be set"

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = getattr(module, "layer_idx", -1)

        k_comp, k_meta = self.key_codec.encode(keys)
        t0 = time.perf_counter()
        keys_hat = self.key_codec.decode(k_comp, k_meta, keys.shape)
        k_decode = time.perf_counter() - t0

        v_comp, v_meta = self.value_codec.encode(values)
        t0 = time.perf_counter()
        values_hat = self.value_codec.decode(v_comp, v_meta, values.shape)
        v_decode = time.perf_counter() - t0

        self.layer_encode_ms[layer_idx] = 0.0
        self.layer_decode_ms[layer_idx] = (k_decode + v_decode) * 1000.0

        k_mse = float((keys.float() - keys_hat.float()).pow(2).mean())
        v_mse = float((values.float() - values_hat.float()).pow(2).mean())
        self.layer_mse[layer_idx] = (k_mse + v_mse) / 2.0

        k_flat = keys.float().reshape(-1)
        kh_flat = keys_hat.float().reshape(-1)
        v_flat = values.float().reshape(-1)
        vh_flat = values_hat.float().reshape(-1)
        k_cs = float((k_flat * kh_flat).sum()) / float((k_flat.norm() * kh_flat.norm()).clamp(min=1e-12))
        v_cs = float((v_flat * vh_flat).sum()) / float((v_flat.norm() * vh_flat.norm()).clamp(min=1e-12))
        self.layer_cosine_sim[layer_idx] = (k_cs + v_cs) / 2.0

        ov = (int(keys.numel()) + int(values.numel())) * 2
        cb = self.key_codec._compressed_bytes(k_comp, k_meta) + self.value_codec._compressed_bytes(v_comp, v_meta)
        self.layer_original_bytes[layer_idx] = ov
        self.layer_compressed_bytes[layer_idx] = min(cb, ov - 1)

        return keys_hat.to(keys.dtype), values_hat.to(values.dtype)

    def summary(self) -> dict:
        total_orig = sum(self.layer_original_bytes.values())
        total_comp = sum(self.layer_compressed_bytes.values())
        eff_ratio = 1.0 - total_comp / max(total_orig, 1)

        avg_mse = sum(self.layer_mse.values()) / max(len(self.layer_mse), 1)
        avg_cs = sum(self.layer_cosine_sim.values()) / max(len(self.layer_cosine_sim), 1)
        avg_encode = sum(self.layer_encode_ms.values()) / max(len(self.layer_encode_ms), 1)
        avg_decode = sum(self.layer_decode_ms.values()) / max(len(self.layer_decode_ms), 1)

        return {
            "effective_compression_ratio": eff_ratio,
            "avg_layer_mse": avg_mse,
            "avg_layer_cosine_sim": avg_cs,
            "avg_layer_encode_ms": avg_encode,
            "avg_layer_decode_ms": avg_decode,
        }

    def reset_stats(self):
        self.layer_encode_ms.clear()
        self.layer_decode_ms.clear()
        self.layer_mse.clear()
        self.layer_cosine_sim.clear()
        self.layer_compressed_bytes.clear()
        self.layer_original_bytes.clear()

    @classmethod
    def from_codec(cls, codec: KVCodec, compression_ratio: float = 0.0) -> "CodecPress":
        return cls(key_codec=codec, value_codec=codec, compression_ratio=compression_ratio)

    @classmethod
    def from_kivi(cls, bits: int = 4, group_size: int = 64, compression_ratio: float = 0.0) -> "CodecPress":
        k_codec = KIVICodec(bits=bits, group_size=group_size, mode="key")
        v_codec = KIVICodec(bits=bits, group_size=group_size, mode="value")
        return cls(key_codec=k_codec, value_codec=v_codec, compression_ratio=compression_ratio)
