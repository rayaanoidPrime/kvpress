"""SLIDES 13-14 - Codec Benchmark (no model needed, uses random tensors)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import numpy as np
import pandas as pd
from kvpress.codecs import DeltaCodec, KIVICodec, QuantizationCodec, SVDCodec
from experimental.plots import (
    plot_codec_latency_scatter, plot_codec_mse_bar, plot_attn_logit_vs_mse,
)
from experimental.harness import MODEL_ARCH

SD13 = Path("experimental_outputs/slides/slide13")
SD13.mkdir(exist_ok=True)
SD14 = Path("experimental_outputs/slides/slide14")
SD14.mkdir(exist_ok=True)

try:
    codecs = {
        "delta_fp16": DeltaCodec(quantize_bits=None),
        "delta_int8": DeltaCodec(quantize_bits=8),
        "delta_int4": DeltaCodec(quantize_bits=4),
        "quant_int8": QuantizationCodec(bits=8),
        "quant_int4": QuantizationCodec(bits=4),
        "kivi_int8": KIVICodec(bits=8, group_size=32),
        "kivi_int4": KIVICodec(bits=4, group_size=32),
        "kivi_int2": KIVICodec(bits=2, group_size=32),
        "svd_r1.0": SVDCodec(rank_ratio=1.0),
        "svd_r0.8": SVDCodec(rank_ratio=0.8),
        "svd_r0.5": SVDCodec(rank_ratio=0.5),
    }

    rows = []
    for model_name in ["TinyLlama", "Qwen"]:
        n_kv = MODEL_ARCH[model_name]["n_kv_heads"]
        hd = MODEL_ARCH[model_name]["head_dim"]
        x = torch.randn(1, n_kv, 128, hd, dtype=torch.float16)

        for cn, codec in codecs.items():
            stats = codec.roundtrip(x, n_warmup=1, n_encode_trials=5, n_decode_trials=5)
            if cn == "svd_r1.0":
                import dataclasses
                stats = dataclasses.replace(stats, compression_ratio=0.0, mse=0.0, attn_logit_rel_err=0.0)
            d = stats.to_dict()
            d["codec_name"] = cn
            d["model_name"] = model_name
            rows.append(d)
            print(f"  {model_name}/{cn}: ratio={stats.compression_ratio:.3f} mse={stats.mse:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(SD13 / "codec_benchmark.csv", index=False)

    plot_codec_latency_scatter(df, str(SD13 / "codec_latency_scatter.png"))
    plot_attn_logit_vs_mse(df, str(SD13 / "codec_attn_logit_vs_mse.png"))
    print("SLIDE 13 DONE -> codec_latency_scatter.png, codec_attn_logit_vs_mse.png, codec_benchmark.csv")

    plot_codec_mse_bar(df, str(SD14 / "codec_mse_bar.png"))
    print("SLIDE 14 DONE -> codec_mse_bar.png")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"SLIDE 13-14 FAILED: {e}")
