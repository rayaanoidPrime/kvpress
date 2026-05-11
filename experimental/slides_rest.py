"""Slides 7-18 continuation."""
import os, sys, traceback, json
from pathlib import Path
import torch, numpy as np, pandas as pd
import torch.nn.functional as F
sys.path.insert(0, str(Path(".").resolve()))

from experimental.harness import MODEL_PATHS, MODEL_ARCH, tokenize, measure_perplexity, needle_in_haystack
from experimental.context_samples import PROSE_CONTEXT, CODE_CONTEXT, NEEDLE_FACTS
from experimental.plots import (
    plot_attention_kl_heatmap, plot_autocorrelation, plot_scree, plot_effective_rank,
    plot_codec_latency_scatter, plot_codec_mse_bar, plot_attn_logit_vs_mse,
    plot_ppl_vs_ratio, plot_ppl_vs_bits, plot_eviction_ppl_sweep,
    plot_crossover_comparison, plot_needle_heatmap,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from kvpress.codecs import DeltaCodec, KIVICodec, QuantizationCodec, SVDCodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress

SLIDE_DIR = Path("experimental_outputs/slides")
INST_DIR = SLIDE_DIR / "instrumented"

def load_model(mn, eager=False):
    path = MODEL_PATHS[mn]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    kw = {"torch_dtype": torch.float16, "device_map": "cpu"}
    if eager:
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    return model, tok

def load_inst(model_name, ctx_type):
    d = INST_DIR / model_name / ctx_type
    st = json.loads((d / "instrumented_stats.json").read_text())
    st = {int(k): v for k, v in st.items()}
    k = torch.load(d / "captured_keys.pt", map_location="cpu", weights_only=False)
    k = {int(kk): vv for kk, vv in k.items()}
    return st, k

# STAGE selects which slides to run
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
print(f"Running stage: {STAGE}")

if STAGE in ("static", "all"):
    print("\n=== SLIDES 10-12 (static) ===")
