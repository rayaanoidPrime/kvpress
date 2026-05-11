"""SLIDE 7 - Attention KL divergence (fixed single-pass approach)."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch, numpy as np, pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from experimental.harness import MODEL_PATHS, MODEL_ARCH, tokenize
from experimental.context_samples import PROSE_CONTEXT
from experimental.plots import plot_attention_kl_heatmap
from kvpress.codecs import DeltaCodec, KIVICodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.knorm_press import KnormPress

SD = Path("experimental_outputs/slides/slide07")
SD.mkdir(exist_ok=True)

def load_model(mn):
    path = MODEL_PATHS[mn]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, device_map="cpu",
        attn_implementation="eager",
    ).eval()
    return model, tok

try:
    kl_rows = []
    kl_dict = {}
    for mn in ["TinyLlama", "Qwen"]:
        print(f"Slide 7: {mn} ...")
        model, tokenizer = load_model(mn)
        t = tokenize(tokenizer, PROSE_CONTEXT, max_length=256)
        sl = t.shape[1]
        n_l = MODEL_ARCH[mn]["n_layers"]
        n_th = model.config.num_attention_heads

        with torch.no_grad():
            out_bl = model(t, output_attentions=True, use_cache=False)
        bl_attn = {i: a.cpu() for i, a in enumerate(out_bl.attentions) if a is not None}
        print(f"  Baseline: {len(bl_attn)} attention layers")

        methods = {
            "delta_int8": CodecPress.from_codec(DeltaCodec(quantize_bits=8)),
            "kivi_int4": CodecPress.from_kivi(bits=4, group_size=32),
            "knorm_0.5": KnormPress(compression_ratio=0.5),
        }
        kl_dict[mn] = {}
        for mname, press in methods.items():
            try:
                print(f"  {mn}/{mname} ...")
                kl_arr = np.full((n_l, n_th), np.nan)
                with torch.no_grad():
                    cache = DynamicCache()
                    with press(model):
                        out_cm = model(
                            t, past_key_values=cache,
                            cache_position=torch.arange(sl, device="cpu"),
                            output_attentions=True, use_cache=True,
                        )
                cm_attn = {i: a.cpu() for i, a in enumerate(out_cm.attentions) if a is not None}

                for layer in range(n_l):
                    if layer in bl_attn and layer in cm_attn:
                        bl = bl_attn[layer]
                        cm = cm_attn[layer]
                        if bl.shape[-1] == cm.shape[-1]:
                            min_heads = min(n_th, bl.shape[1], cm.shape[1])
                            for head in range(min_heads):
                                try:
                                    kl_val = F.kl_div(
                                        F.log_softmax(cm[0, head].float(), dim=-1),
                                        F.softmax(bl[0, head].float(), dim=-1),
                                        reduction="batchmean",
                                    ).item()
                                    kl_arr[layer, head] = kl_val
                                except Exception:
                                    pass

                kl_dict[mn][mname] = kl_arr
                for layer in range(n_l):
                    for head in range(n_th):
                        kl_rows.append({
                            "model": mn, "method": mname, "layer": layer,
                            "head": head, "kl_divergence": float(kl_arr[layer, head]),
                        })
                valid = ~np.isnan(kl_arr)
                mean_kl = float(np.mean(kl_arr[valid])) if valid.any() else float("nan")
                n_compared = int(valid.any(axis=1).sum())
                print(f"  {mn}/{mname}: layers_compared={n_compared} mean_KL={mean_kl:.6f}")
            except Exception as e2:
                print(f"  {mn}/{mname} FAILED: {e2}")
                traceback.print_exc()
                kl_dict[mn][mname] = np.full((n_l, n_th), np.nan)
        del model

    kl_df = pd.DataFrame(kl_rows)
    kl_df.to_csv(SD / "attention_kl.csv", index=False)
    plot_attention_kl_heatmap(kl_dict, str(SD / "attention_kl_heatmap.png"))
    print("SLIDE 07 DONE -> attention_kl_heatmap.png, attention_kl.csv")
except Exception as e:
    traceback.print_exc()
    print(f"SLIDE 07 FAILED: {e}")
