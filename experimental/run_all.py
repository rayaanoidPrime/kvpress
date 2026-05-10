# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import traceback
from datetime import datetime
from pathlib import Path

from experimental.experiments import (
    OUTPUT_ROOT,
    run_codec_benchmark,
    run_crossover_comparison,
    run_instrumented_press,
    run_needle_sweep,
    run_ppl_codec_sweep,
    run_ppl_eviction_sweep,
)

STAGES = [
    "codec_benchmark",
    "instrumented_press",
    "ppl_codec_sweep",
    "ppl_eviction_sweep",
    "crossover_comparison",
    "needle_sweep",
]


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    for stage in STAGES:
        out_dir = OUTPUT_ROOT / f"{stage}_{_timestamp()}"
        try:
            if stage == "codec_benchmark":
                for mn in ["TinyLlama", "Qwen"]:
                    d = out_dir / mn
                    d.mkdir(parents=True, exist_ok=True)
                    run_codec_benchmark(model_name=mn, output_dir=str(d))
                print(f"DONE: codec_benchmark -> {out_dir}")

            elif stage == "instrumented_press":
                for mn in ["TinyLlama", "Qwen"]:
                    for ctx in ["prose", "code"]:
                        d = out_dir / mn / ctx
                        d.mkdir(parents=True, exist_ok=True)
                        run_instrumented_press(model_name=mn, context_type=ctx, output_dir=str(d))
                print(f"DONE: instrumented_press -> {out_dir}")

            elif stage == "ppl_codec_sweep":
                for mn in ["TinyLlama", "Qwen"]:
                    for ctx in ["prose", "code"]:
                        d = out_dir / mn / ctx
                        d.mkdir(parents=True, exist_ok=True)
                        run_ppl_codec_sweep(model_name=mn, context_type=ctx, output_dir=str(d))
                print(f"DONE: ppl_codec_sweep -> {out_dir}")

            elif stage == "ppl_eviction_sweep":
                for mn in ["TinyLlama", "Qwen"]:
                    for ctx in ["prose", "code"]:
                        d = out_dir / mn / ctx
                        d.mkdir(parents=True, exist_ok=True)
                        run_ppl_eviction_sweep(model_name=mn, context_type=ctx, output_dir=str(d))
                print(f"DONE: ppl_eviction_sweep -> {out_dir}")

            elif stage == "crossover_comparison":
                for mn in ["TinyLlama", "Qwen"]:
                    d = out_dir / mn
                    d.mkdir(parents=True, exist_ok=True)
                    run_crossover_comparison(model_name=mn, context_type="prose", output_dir=str(d))
                print(f"DONE: crossover_comparison -> {out_dir}")

            elif stage == "needle_sweep":
                for mn in ["TinyLlama", "Qwen"]:
                    d = out_dir / mn
                    d.mkdir(parents=True, exist_ok=True)
                    run_needle_sweep(model_name=mn, output_dir=str(d))
                print(f"DONE: needle_sweep -> {out_dir}")

        except Exception:
            traceback.print_exc()
            print(f"FAILED: {stage}")


if __name__ == "__main__":
    main()
