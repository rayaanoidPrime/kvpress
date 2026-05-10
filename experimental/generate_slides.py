# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os, sys, traceback, json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).parent))

from experimental.harness import MODEL_PATHS, MODEL_ARCH, tokenize
from experimental.context_samples import PROSE_CONTEXT, CODE_CONTEXT, NEEDLE_FACTS
from experimental.plots import (
    plot_kv_growth, plot_magnitude_heatmap, plot_kv_distributions,
    plot_channel_variance, plot_token_variance, plot_layer_depth_profiles,
    plot_attention_kl_heatmap, plot_autocorrelation,
    plot_scree, plot_effective_rank, plot_codec_latency_scatter,
    plot_codec_mse_bar, plot_attn_logit_vs_mse, plot_ppl_vs_ratio,
    plot_ppl_vs_bits, plot_eviction_ppl_sweep, plot_crossover_comparison,
    plot_needle_heatmap,
)
from kvpress.codecs import DeltaCodec, KIVICodec, QuantizationCodec, SVDCodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.instrumented_press import InstrumentedPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress

SLIDE_DIR = Path("experimental_outputs/slides")
INST_DIR = SLIDE_DIR / "instrumented"
COMPLETED = []
FAILED = []
FILES_SAVED = {}

def slide_done(n, files):
    COMPLETED.append(n)
    FILES_SAVED[n] = files
    print(f"SLIDE {n:02d} DONE -> {files}")

def slide_fail(n, exc):
    msg = str(exc).split("\n")[0]
    FAILED.append((n, msg))
    traceback.print_exc()
    print(f"SLIDE {n:02d} FAILED -> {msg}")

def load_model(model_name, eager=False):
    path = MODEL_PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {"torch_dtype": torch.float16, "device_map": "cpu"}
    if eager:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    return model, tokenizer

print("Header OK, beginning slides...")

# ============= SLIDE 1 =============
try:
    sd = SLIDE_DIR / "slide01"
    sd.mkdir(exist_ok=True)
    info = {
        "TinyLlama": {"n_layers": 22, "n_kv_heads": 4, "head_dim": 64},
        "Qwen": {"n_layers": 24, "n_kv_heads": 8, "head_dim": 64},
    }
    seqs = [128, 256, 512, 1024, 2048, 4096, 8192]
    rows = []
    for mn, a in info.items():
        for sl in seqs:
            mb = 2 * a["n_layers"] * a["n_kv_heads"] * a["head_dim"] * sl * 2 / 1e6
            rows.append({"model_name": mn, "seq_len": sl, "kv_cache_mb": mb})
    df1 = pd.DataFrame(rows)
    df1.to_csv(sd / "kv_growth.csv", index=False)
    plot_kv_growth(df1, str(sd / "kv_growth.png"))
    slide_done(1, ["kv_growth.png", "kv_growth.csv"])
except Exception as e:
    slide_fail(1, e)

# ============= SLIDES 2-6 PREP: Run InstrumentedPress =============
print("--- Running InstrumentedPress for all models/contexts ---")
for model_name in ["TinyLlama", "Qwen"]:
    for ctx_type in ["prose", "code"]:
        try:
            out_dir = INST_DIR / model_name / ctx_type
            out_dir.mkdir(parents=True, exist_ok=True)
            model, tokenizer = load_model(model_name)
            context = PROSE_CONTEXT if ctx_type == "prose" else CODE_CONTEXT
            input_ids = tokenize(tokenizer, context, max_length=512)
            seq_len = input_ids.shape[1]
            press = InstrumentedPress()
            with torch.no_grad():
                with press(model):
                    cache = DynamicCache()
                    model(input_ids, past_key_values=cache,
                          cache_position=torch.arange(seq_len, device="cpu"), use_cache=True)
            press.save(str(out_dir))
            del model
            print(f"  INSTRUMENTED: {model_name}/{ctx_type} saved")
        except Exception as e:
            print(f"  INSTRUMENTED FAILED: {model_name}/{ctx_type}: {e}")
            traceback.print_exc()

def load_inst(model_name, ctx_type):
    d = INST_DIR / model_name / ctx_type
    with open(d / "instrumented_stats.json") as f:
        st = json.load(f)
    st = {int(k): v for k, v in st.items()}
    k = torch.load(d / "captured_keys.pt", map_location="cpu", weights_only=False)
    v = torch.load(d / "captured_values.pt", map_location="cpu", weights_only=False)
    k = {int(kk): vv for kk, vv in k.items()}
    v = {int(kk): vv for kk, vv in v.items()}
    return st, k, v

# ============= SLIDE 2 =============
try:
    sd = SLIDE_DIR / "slide02"
    sd.mkdir(exist_ok=True)
    for ctx in ["prose", "code"]:
        ckd = {}
        for mn in ["TinyLlama", "Qwen"]:
            _, kd, _ = load_inst(mn, ctx)
            ckd[mn] = kd
        mid = MODEL_ARCH["TinyLlama"]["n_layers"] // 2
        plot_magnitude_heatmap(ckd, None, str(sd / f"magnitude_heatmap_{ctx}.png"), mid, ctx)
    slide_done(2, ["magnitude_heatmap_prose.png", "magnitude_heatmap_code.png"])
except Exception as e:
    slide_fail(2, e)

# ============= SLIDE 3 =============
try:
    sd = SLIDE_DIR / "slide03"
    sd.mkdir(exist_ok=True)
    for ctx in ["prose", "code"]:
        ck, cv = {}, {}
        for mn in ["TinyLlama", "Qwen"]:
            _, kd, vd = load_inst(mn, ctx)
            ck[mn] = kd
            cv[mn] = vd
        mid = MODEL_ARCH["TinyLlama"]["n_layers"] // 2
        plot_kv_distributions(ck, cv, str(sd / f"kv_distributions_{ctx}.png"), ["TinyLlama", "Qwen"], mid)
    slide_done(3, ["kv_distributions_prose.png", "kv_distributions_code.png"])
except Exception as e:
    slide_fail(3, e)

# ============= SLIDE 4 =============
try:
    sd = SLIDE_DIR / "slide04"
    sd.mkdir(exist_ok=True)
    for ctx in ["prose", "code"]:
        isd = {}
        for mn in ["TinyLlama", "Qwen"]:
            st, _, _ = load_inst(mn, ctx)
            isd[mn] = st
        plot_channel_variance(isd, str(sd / f"channel_variance_{ctx}.png"), ctx)
    slide_done(4, ["channel_variance_prose.png", "channel_variance_code.png"])
except Exception as e:
    slide_fail(4, e)

# ============= SLIDE 5 =============
try:
    sd = SLIDE_DIR / "slide05"
    sd.mkdir(exist_ok=True)
    for ctx in ["prose", "code"]:
        isd = {}
        for mn in ["TinyLlama", "Qwen"]:
            st, _, _ = load_inst(mn, ctx)
            isd[mn] = st
        plot_token_variance(isd, str(sd / f"token_variance_{ctx}.png"), ctx)
    slide_done(5, ["token_variance_prose.png", "token_variance_code.png"])
except Exception as e:
    slide_fail(5, e)

# ============= SLIDE 6 =============
try:
    sd = SLIDE_DIR / "slide06"
    sd.mkdir(exist_ok=True)
    dfs = []
    for mn in ["TinyLlama", "Qwen"]:
        for ctx in ["prose", "code"]:
            csv_path = INST_DIR / mn / ctx / "layer_stats_summary.csv"
            sub = pd.read_csv(csv_path)
            sub["model_name"] = mn
            sub["context_type"] = ctx
            dfs.append(sub)
    lsd = pd.concat(dfs, ignore_index=True)
    lsd.to_csv(sd / "layer_stats_combined.csv", index=False)
    plot_layer_depth_profiles(lsd, str(sd / "layer_depth_profiles.png"))
    slide_done(6, ["layer_depth_profiles.png", "layer_stats_combined.csv"])
except Exception as e:
    slide_fail(6, e)

print(f"Slides 1-6 complete. {len(COMPLETED)} passed, {len(FAILED)} failed")

# ============= SLIDE 7 =============
try:
    sd = SLIDE_DIR / "slide07"
    sd.mkdir(exist_ok=True)
    kl_rows = []
    kl_dict = {}
    for mn in ["TinyLlama", "Qwen"]:
        print(f"  Slide7: {mn} attention KL ...")
        model, tokenizer = load_model(mn, eager=True)
        input_ids = tokenize(tokenizer, PROSE_CONTEXT, max_length=256)
        slen = input_ids.shape[1]
        n_l = MODEL_ARCH[mn]["n_layers"]
        n_total_heads = model.config.num_attention_heads
        with torch.no_grad():
            out_bl = model(input_ids, output_attentions=True, use_cache=False)
        bl_attn = {i: a.cpu() for i, a in enumerate(out_bl.attentions) if a is not None}
        methods = {
            "delta_int8": CodecPress.from_codec(DeltaCodec(quantize_bits=8)),
            "kivi_int4": CodecPress.from_kivi(bits=4, group_size=32),
            "knorm_0.5": KnormPress(compression_ratio=0.5),
        }
        kl_dict[mn] = {}
        for mname, press in methods.items():
            try:
                kl_arr = np.full((n_l, n_total_heads), np.nan)
                with torch.no_grad():
                    cache = DynamicCache()
                    with press(model):
                        model(input_ids, past_key_values=cache,
                              cache_position=torch.arange(slen, device="cpu"), use_cache=True)
                    out_cm = model(input_ids, past_key_values=cache,
                                   output_attentions=True, use_cache=False)
                cm_attn = {i: a.cpu() for i, a in enumerate(out_cm.attentions) if a is not None}
                for layer in range(n_l):
                    if layer in bl_attn and layer in cm_attn:
                        bl = bl_attn[layer]
                        cm = cm_attn[layer]
                        for head in range(min(n_total_heads, bl.shape[1], cm.shape[1])):
                            try:
                                if bl.shape[2] == cm.shape[2]:
                                    kl_val = F.kl_div(
                                        F.log_softmax(cm[0, head].float(), dim=-1),
                                        F.softmax(bl[0, head].float(), dim=-1),
                                        reduction="batchmean",
                                    ).item()
                                    kl_arr[layer, head] = kl_val
                            except:
                                pass
                kl_dict[mn][mname] = kl_arr
                for layer in range(n_l):
                    for head in range(n_total_heads):
                        kl_rows.append({
                            "model": mn, "method": mname, "layer": layer,
                            "head": head, "kl_divergence": float(kl_arr[layer, head]),
                        })
            except Exception as e2:
                print(f"    WARN {mn} {mname}: {e2}")
                kl_dict[mn][mname] = np.full((n_l, n_total_heads), np.nan)
        del model
    kl_df = pd.DataFrame(kl_rows)
    kl_df.to_csv(sd / "attention_kl.csv", index=False)
    plot_attention_kl_heatmap(kl_dict, str(sd / "attention_kl_heatmap.png"))
    slide_done(7, ["attention_kl_heatmap.png", "attention_kl.csv"])
except Exception as e:
    slide_fail(7, e)

print(f"After slide 7: {len(COMPLETED)} passed, {len(FAILED)} failed")

# ============= SLIDE 8 =============
try:
    sd = SLIDE_DIR / "slide08"
    sd.mkdir(exist_ok=True)

    import matplotlib as mpl_backend
    mpl_backend.use("Agg")
    import matplotlib.pyplot as plt

    class RecKnPress(KnormPress):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.indices = {}
        def compress(self, mod, hs, keys, vals, attns, kw):
            li = getattr(mod, "layer_idx", len(self.indices))
            kl = keys.shape[2]
            if self.compression_ratio == 0:
                self.indices[li] = torch.arange(kl)
                return keys, vals
            scores = self.score(mod, hs, keys, vals, attns, kw)
            nk = int(kl * (1 - self.compression_ratio))
            idx = scores.topk(nk, dim=-1).indices
            self.indices[li] = idx
            ie = idx.unsqueeze(-1).expand(-1, -1, -1, mod.head_dim)
            return keys.gather(2, ie).contiguous(), vals.gather(2, ie).contiguous()

    class RecSnapPress(SnapKVPress):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.indices = {}
        def compress(self, mod, hs, keys, vals, attns, kw):
            li = getattr(mod, "layer_idx", len(self.indices))
            kl = keys.shape[2]
            if self.compression_ratio == 0:
                self.indices[li] = torch.arange(kl)
                return keys, vals
            scores = self.score(mod, hs, keys, vals, attns, kw)
            nk = int(kl * (1 - self.compression_ratio))
            idx = scores.topk(nk, dim=-1).indices
            self.indices[li] = idx
            ie = idx.unsqueeze(-1).expand(-1, -1, -1, mod.head_dim)
            return keys.gather(2, ie).contiguous(), vals.gather(2, ie).contiguous()

    for model_name in ["TinyLlama", "Qwen"]:
        print(f"  Slide8: {model_name} eviction ...")
        model, tokenizer = load_model(model_name)
        results = []
        for press_cls, pn in [(RecKnPress, "KNorm"), (RecSnapPress, "SnapKV")]:
            row = []
            for ctx_name, ctx_text in [("prose", PROSE_CONTEXT), ("code", CODE_CONTEXT)]:
                t = tokenize(tokenizer, ctx_text, max_length=512)
                sl = t.shape[1]
                pr = press_cls(compression_ratio=0.5)
                with torch.no_grad():
                    with pr(model):
                        cache = DynamicCache()
                        model(t, past_key_values=cache,
                              cache_position=torch.arange(sl), use_cache=True)
                masks = []
                for li in sorted(pr.indices.keys()):
                    idx = pr.indices[li][0, 0]
                    mk = torch.zeros(sl)
                    mk[idx] = 1.0
                    masks.append(mk)
                avg_mask = torch.stack(masks).mean(dim=0)
                token_strs = tokenizer.convert_ids_to_tokens(t[0].tolist())
                row.append((token_strs, avg_mask.tolist()))
            results.append(row)
        # 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(18, max(6, 4 * 2 * 1.2)), facecolor="white")
        methods = ["KNorm (CR=0.5)", "SnapKV (CR=0.5)"]
        ctx_labels = ["prose", "code"]
        ch = "#2563EB"
        cm = "#D97706"
        cl = "#DC2626"
        for ri in range(2):
            for ci in range(2):
                ax = axes[ri, ci]
                toks, prbs = results[ri][ci]
                nt = len(toks)
                pr = 80
                nr = max(1, int(np.ceil(nt / pr)))
                for i, (tk, pb) in enumerate(zip(toks, prbs)):
                    rw = i // pr
                    cc = i % pr
                    if pb >= 0.8:
                        col = ch
                    elif pb >= 0.4:
                        col = cm
                    else:
                        col = cl
                    ax.text(cc, nr - rw - 1, tk, fontfamily="monospace", fontsize=5,
                            color=col, ha="center", va="center")
                ax.set_xlim(-0.5, pr - 0.5)
                ax.set_ylim(-0.5, nr - 0.5)
                ax.set_title(f"{methods[ri]} x {ctx_labels[ci]}")
                ax.axis("off")
        fig.suptitle(f"Eviction Pattern - {model_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(sd / f"eviction_pattern_{model_name.lower()}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        del model
    slide_done(8, ["eviction_pattern_tinyllama.png", "eviction_pattern_qwen.png"])
except Exception as e:
    slide_fail(8, e)

# ============= SLIDE 9A =============
try:
    sd = SLIDE_DIR / "slide09a"
    sd.mkdir(exist_ok=True)
    ac_rows = []
    ac_dict = {}
    for mn in ["TinyLlama", "Qwen"]:
        ac_dict[mn] = {}
        for ctx in ["prose", "code"]:
            st, _, _ = load_inst(mn, ctx)
            layers = sorted(st.keys())
            e = layers[0]
            m = layers[len(layers) // 2]
            l = layers[-1]
            ac_dict[mn][ctx] = {
                "early": st[e]["autocorr_lags_1_to_20"],
                "middle": st[m]["autocorr_lags_1_to_20"],
                "late": st[l]["autocorr_lags_1_to_20"],
            }
            for dp, li in [("early", e), ("middle", m), ("late", l)]:
                for i, val in enumerate(st[li]["autocorr_lags_1_to_20"]):
                    ac_rows.append({
                        "model": mn, "context": ctx, "depth": dp,
                        "layer": li, "lag": i + 1, "cosine_sim": val,
                    })
    pd.DataFrame(ac_rows).to_csv(sd / "autocorrelation.csv", index=False)
    plot_autocorrelation(ac_dict, str(sd / "autocorrelation.png"))
    slide_done(9, ["autocorrelation.png", "autocorrelation.csv"])
except Exception as e:
    slide_fail(9, e)

# ============= SLIDE 9B =============
try:
    sd = SLIDE_DIR / "slide09b"
    sd.mkdir(exist_ok=True)
    sc_dict = {}
    er_rows = []
    for mn in ["TinyLlama", "Qwen"]:
        sc_dict[mn] = {}
        for ctx in ["prose", "code"]:
            st, _, _ = load_inst(mn, ctx)
            for li in sorted(st.keys()):
                sc_dict[mn][li] = st[li]["sv_cumvar"]
                er_rows.append({
                    "model_name": mn, "context_type": ctx,
                    "layer_idx": li, "effective_rank_90": st[li]["effective_rank_90"],
                })
    er_df = pd.DataFrame(er_rows)
    er_df.to_csv(sd / "effective_rank.csv", index=False)
    plot_scree(sc_dict, str(sd / "scree_plot.png"))
    plot_effective_rank(er_df, str(sd / "effective_rank_bar.png"))
    slide_done(10, ["scree_plot.png", "effective_rank_bar.png", "effective_rank.csv"])
except Exception as e:
    slide_fail(10, e)

print(f"After slides 8-10: {len(COMPLETED)} passed, {len(FAILED)} failed")

print("=" * 50)
print(f"COMPLETED: {len(COMPLETED)}, FAILED: {len(FAILED)}")
for n in sorted(COMPLETED):
    print(f"  Slide {n:02d}: OK - {FILES_SAVED.get(n, [])}")
for n, msg in sorted(FAILED, key=lambda x: x[0]):
    print(f"  Slide {n:02d}: FAIL - {msg}")
