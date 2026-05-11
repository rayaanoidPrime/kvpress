"""SLIDE 15 - PPL Codec + Eviction Sweep (needs model, heavy)."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from experimental.harness import MODEL_PATHS, MODEL_ARCH, tokenize, measure_perplexity
from experimental.context_samples import PROSE_CONTEXT, CODE_CONTEXT
from experimental.plots import plot_ppl_vs_ratio, plot_eviction_ppl_sweep
from kvpress.codecs import DeltaCodec, KIVICodec, QuantizationCodec, SVDCodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress

SD = Path("experimental_outputs/slides/slide15")
SD.mkdir(exist_ok=True)

def load_model(mn):
    path = MODEL_PATHS[mn]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="cpu",
    ).eval()
    return model, tok

try:
    codec_configs = {
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

    eviction_configs = {
        "knorm": KnormPress,
        "snapkv": SnapKVPress,
    }
    eviction_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    all_rows = []
    for model_name in ["TinyLlama", "Qwen"]:
        print(f"Loading {model_name} ...")
        model, tokenizer = load_model(model_name)
        for ctx_type, ctx_text in [("prose", PROSE_CONTEXT), ("code", CODE_CONTEXT)]:
            print(f"  {model_name}/{ctx_type} codec sweep ...")
            for cn, press in codec_configs.items():
                result = measure_perplexity(model, tokenizer, ctx_text, press, max_length=512)
                eff_cr = 0.0
                if press is not None and hasattr(press, "summary"):
                    s = press.summary()
                    eff_cr = s.get("effective_compression_ratio", 0.0)
                all_rows.append({
                    "codec_name": cn, "model_name": model_name, "context_type": ctx_type,
                    "perplexity": result["perplexity"], "loss": result["loss"],
                    "inference_ms": result["inference_ms"],
                    "effective_compression_ratio": eff_cr,
                })
                print(f"    {cn}: ppl={result['perplexity']:.2f} cr={eff_cr:.3f}")

            # Eviction sweep for prose only
            if ctx_type == "prose":
                print(f"  {model_name}/prose eviction sweep ...")
                ev_rows = []
                for ratio in eviction_ratios:
                    for pn, pcls in eviction_configs.items():
                        press = pcls(compression_ratio=ratio)
                        result = measure_perplexity(model, tokenizer, ctx_text, press, max_length=512)
                        ev_rows.append({
                            "press_name": pn, "compression_ratio": ratio,
                            "model_name": model_name, "context_type": ctx_type,
                            "perplexity": result["perplexity"], "loss": result["loss"],
                            "inference_ms": result["inference_ms"],
                        })
                ev_df = pd.DataFrame(ev_rows)
                plot_eviction_ppl_sweep(ev_df, str(SD / f"ppl_eviction_{model_name.lower()}_prose.png"))
                print(f"    eviction plot saved")
        del model

    df = pd.DataFrame(all_rows)
    df.to_csv(SD / "ppl_codec_sweep_all.csv", index=False)

    for mn in ["TinyLlama", "Qwen"]:
        for ctx in ["prose", "code"]:
            sub = df[(df["model_name"] == mn) & (df["context_type"] == ctx)]
            plot_ppl_vs_ratio(sub, str(SD / f"ppl_vs_ratio_{mn.lower()}_{ctx}.png"),
                             f"{mn} - {ctx}")

    print("SLIDE 15 DONE -> ppl_codec_sweep_all.csv + 6 plots")
except Exception as e:
    traceback.print_exc()
    print(f"SLIDE 15 FAILED: {e}")
