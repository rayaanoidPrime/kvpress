"""Slides 2,7,8,9a,9b,10,11,12,13,14,15,16,17,18 continuation"""
import os, sys, traceback, json
from pathlib import Path
import torch, numpy as np, pandas as pd
import torch.nn.functional as F
sys.path.insert(0, str(Path(".").resolve()))
from experimental.harness import MODEL_PATHS, MODEL_ARCH, tokenize
from experimental.context_samples import PROSE_CONTEXT, CODE_CONTEXT, NEEDLE_FACTS
from experimental.plots import (
    plot_magnitude_heatmap, plot_attention_kl_heatmap, plot_autocorrelation,
    plot_scree, plot_effective_rank,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from kvpress.codecs import DeltaCodec, KIVICodec, QuantizationCodec, SVDCodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress

SLIDE_DIR = Path("experimental_outputs/slides")
INST_DIR = SLIDE_DIR / "instrumented"

def load_inst(model_name, ctx_type):
    d = INST_DIR / model_name / ctx_type
    st = json.loads((d / "instrumented_stats.json").read_text())
    st = {int(k): v for k, v in st.items()}
    k = torch.load(d / "captured_keys.pt", map_location="cpu", weights_only=False)
    v = torch.load(d / "captured_values.pt", map_location="cpu", weights_only=False)
    k = {int(kk): vv for kk, vv in k.items()}
    v = {int(kk): vv for kk, vv in v.items()}
    return st, k, v

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

# SLIDE 2 fixed: build full 4-key dict
print("=== SLIDE 2 FIX ===")
sd = SLIDE_DIR / "slide02"
sd.mkdir(exist_ok=True)
ckd = {}
for ctx in ["prose", "code"]:
    for mn in ["TinyLlama", "Qwen"]:
        _, kd, _ = load_inst(mn, ctx)
        ckd[(mn, ctx)] = kd
mid = MODEL_ARCH["TinyLlama"]["n_layers"] // 2
plot_magnitude_heatmap(ckd, None, str(sd / "magnitude_heatmap_prose.png"), mid, "prose")
print("SLIDE 02 DONE")
