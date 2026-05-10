# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import torch

from experimental.context_samples import CODE_CONTEXT, NEEDLE_FACTS, PROSE_CONTEXT
from experimental.harness import (
    MODEL_ARCH,
    load_model,
    measure_perplexity,
    needle_in_haystack,
)
from kvpress.codecs import DeltaCodec, KIVICodec, QuantizationCodec, SVDCodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.instrumented_press import InstrumentedPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress
from transformers import DynamicCache

OUTPUT_ROOT = Path("experimental_outputs")


def run_codec_benchmark(
    model_name: str = "TinyLlama",
    seq_len: int = 256,
    n_trials: int = 20,
    output_dir: str = None,
) -> pd.DataFrame:
    n_kv_heads = MODEL_ARCH[model_name]["n_kv_heads"]
    head_dim = MODEL_ARCH[model_name]["head_dim"]
    x = torch.randn(1, n_kv_heads, seq_len, head_dim, dtype=torch.float16)

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
    for codec_name, codec in codecs.items():
        stats = codec.roundtrip(x, n_decode_trials=n_trials)
        if codec_name == "svd_r1.0":
            import dataclasses
            stats = dataclasses.replace(
                stats, compression_ratio=0.0, mse=0.0, attn_logit_rel_err=0.0
            )
        d = stats.to_dict()
        d["codec_name"] = codec_name
        d["model_name"] = model_name
        rows.append(d)

    df = pd.DataFrame(rows)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / "codec_benchmark.csv", index=False)
    return df


def run_instrumented_press(
    model_name: str,
    context_type: str,
    max_length: int = 512,
    output_dir: str = None,
) -> InstrumentedPress:
    model, tokenizer = load_model(model_name)
    context = PROSE_CONTEXT if context_type == "prose" else CODE_CONTEXT
    from experimental.harness import tokenize
    input_ids = tokenize(tokenizer, context, max_length=max_length)
    seq_len = input_ids.shape[1]

    press = InstrumentedPress()
    with torch.no_grad():
        with press(model):
            cache = DynamicCache()
            model(
                input_ids,
                past_key_values=cache,
                cache_position=torch.arange(seq_len, device="cpu"),
                use_cache=True,
            )
    if output_dir is not None:
        press.save(output_dir)
    return press


def run_ppl_codec_sweep(
    model_name: str,
    context_type: str,
    max_length: int = 512,
    output_dir: str = None,
) -> pd.DataFrame:
    model, tokenizer = load_model(model_name)
    context = PROSE_CONTEXT if context_type == "prose" else CODE_CONTEXT

    configs = {
        "baseline": None,
        "delta_fp16": CodecPress.from_codec(DeltaCodec(quantize_bits=None)),
        "delta_int8": CodecPress.from_codec(DeltaCodec(quantize_bits=8)),
        "delta_int4": CodecPress.from_codec(DeltaCodec(quantize_bits=4)),
        "quant_int8": CodecPress.from_codec(QuantizationCodec(bits=8)),
        "quant_int4": CodecPress.from_codec(QuantizationCodec(bits=4)),
        "kivi_int8": CodecPress.from_kivi(bits=8, group_size=32),
        "kivi_int4": CodecPress.from_kivi(bits=4, group_size=32),
        "kivi_int2": CodecPress.from_kivi(bits=2, group_size=32),
        "svd_r0.8": CodecPress.from_codec(SVDCodec(rank_ratio=0.8)),
        "svd_r0.5": CodecPress.from_codec(SVDCodec(rank_ratio=0.5)),
    }

    rows = []
    for codec_name, press in configs.items():
        result = measure_perplexity(model, tokenizer, context, press, max_length)
        effective_cr = 0.0
        if press is not None and isinstance(press, CodecPress):
            s = press.summary()
            effective_cr = s.get("effective_compression_ratio", 0.0)
        row = {
            "codec_name": codec_name,
            "model_name": model_name,
            "context_type": context_type,
            "perplexity": result["perplexity"],
            "loss": result["loss"],
            "inference_ms": result["inference_ms"],
            "avg_kv_entries": result["avg_kv_entries"],
            "effective_compression_ratio": effective_cr,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / "ppl_codec_sweep.csv", index=False)
    return df


def run_ppl_eviction_sweep(
    model_name: str,
    context_type: str,
    ratios: list = None,
    max_length: int = 512,
    output_dir: str = None,
) -> pd.DataFrame:
    if ratios is None:
        ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    model, tokenizer = load_model(model_name)
    context = PROSE_CONTEXT if context_type == "prose" else CODE_CONTEXT

    rows = []
    for ratio in ratios:
        for press_cls, press_name in [(KnormPress, "knorm"), (SnapKVPress, "snapkv")]:
            press = press_cls(compression_ratio=ratio)
            result = measure_perplexity(model, tokenizer, context, press, max_length)
            rows.append({
                "press_name": press_name,
                "compression_ratio": ratio,
                "model_name": model_name,
                "context_type": context_type,
                "perplexity": result["perplexity"],
                "loss": result["loss"],
                "inference_ms": result["inference_ms"],
            })

    df = pd.DataFrame(rows)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / "ppl_eviction_sweep.csv", index=False)
    return df


def run_crossover_comparison(
    model_name: str,
    context_type: str = "prose",
    max_length: int = 512,
    output_dir: str = None,
) -> pd.DataFrame:
    model, tokenizer = load_model(model_name)
    context = PROSE_CONTEXT if context_type == "prose" else CODE_CONTEXT

    baseline_result = measure_perplexity(model, tokenizer, context, None, max_length)
    baseline_ppl = baseline_result["perplexity"]

    configs = []
    configs.append(("delta_int8", "2x", "precision", CodecPress.from_codec(DeltaCodec(quantize_bits=8))))
    configs.append(("quant_int8", "2x", "precision", CodecPress.from_codec(QuantizationCodec(bits=8))))
    configs.append(("kivi_int4", "2x", "precision", CodecPress.from_kivi(bits=4, group_size=32)))
    configs.append(("svd_r0.5", "2x", "dimension", CodecPress.from_codec(SVDCodec(rank_ratio=0.5))))
    configs.append(("knorm_0.5", "2x", "eviction", KnormPress(compression_ratio=0.5)))
    configs.append(("snapkv_0.5", "2x", "eviction", SnapKVPress(compression_ratio=0.5)))

    configs.append(("delta_int4", "4x", "precision", CodecPress.from_codec(DeltaCodec(quantize_bits=4))))
    configs.append(("quant_int4", "4x", "precision", CodecPress.from_codec(QuantizationCodec(bits=4))))
    configs.append(("kivi_int2", "4x", "precision", CodecPress.from_kivi(bits=2, group_size=32)))
    configs.append(("svd_r0.25", "4x", "dimension", CodecPress.from_codec(SVDCodec(rank_ratio=0.25))))
    configs.append(("knorm_0.75", "4x", "eviction", KnormPress(compression_ratio=0.75)))
    configs.append(("snapkv_0.75", "4x", "eviction", SnapKVPress(compression_ratio=0.75)))

    rows = []
    for method, target, axis, press in configs:
        result = measure_perplexity(model, tokenizer, context, press, max_length)
        rows.append({
            "method": method,
            "target_reduction": target,
            "compression_axis": axis,
            "perplexity": result["perplexity"],
            "delta_ppl": result["perplexity"] - baseline_ppl,
            "model_name": model_name,
            "context_type": context_type,
        })

    df = pd.DataFrame(rows)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(output_dir) / "crossover_comparison.csv", index=False)
    return df


def run_needle_sweep(
    model_name: str = "TinyLlama",
    positions: list = None,
    runs_per_cell: int = 3,
    max_context_tokens: int = 512,
    output_dir: str = None,
) -> pd.DataFrame:
    if positions is None:
        positions = [0.1, 0.3, 0.5, 0.7, 0.9]

    model, tokenizer = load_model(model_name)

    press_grid = {
        "baseline": None,
        "knorm_0.5": KnormPress(compression_ratio=0.5),
        "snapkv_0.5": SnapKVPress(compression_ratio=0.5),
        "delta_int8": CodecPress.from_codec(DeltaCodec(quantize_bits=8)),
        "kivi_int4": CodecPress.from_kivi(bits=4, group_size=32),
    }

    rows = []
    for press_name, press in press_grid.items():
        for pos in positions:
            for run_idx in range(runs_per_cell):
                needle_fact, question, expected_answer = NEEDLE_FACTS[run_idx % len(NEEDLE_FACTS)]
                result = needle_in_haystack(
                    model, tokenizer,
                    haystack_text=PROSE_CONTEXT,
                    needle_fact=needle_fact,
                    question=question,
                    expected_answer=expected_answer,
                    needle_position=pos,
                    max_context_tokens=max_context_tokens,
                    press=press,
                )
                rows.append({
                    "press_name": press_name,
                    "needle_position": pos,
                    "run_idx": run_idx,
                    "needle_fact": needle_fact,
                    "exact_match": result["exact_match"],
                    "f1": result["f1"],
                    "answer": result["answer"],
                    "context_tokens": result["context_tokens"],
                    "inference_ms": result["inference_ms"],
                })

    raw_df = pd.DataFrame(rows)
    agg = raw_df.groupby(["press_name", "needle_position"]).agg(
        mean_exact_match=("exact_match", "mean"),
        mean_f1=("f1", "mean"),
    ).reset_index()

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(Path(output_dir) / "needle_raw.csv", index=False)
        agg.to_csv(Path(output_dir) / "needle_aggregated.csv", index=False)
    return agg
