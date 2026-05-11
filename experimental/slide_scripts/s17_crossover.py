"""SLIDE 17 - Crossover Comparison (needs model)."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import pandas as pd
from experimental.harness import MODEL_PATHS, measure_perplexity
from experimental.context_samples import PROSE_CONTEXT
from experimental.plots import plot_crossover_comparison
from kvpress.codecs import DeltaCodec, KIVICodec, QuantizationCodec, SVDCodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress
from transformers import AutoModelForCausalLM, AutoTokenizer

SD = Path("experimental_outputs/slides/slide17")
SD.mkdir(exist_ok=True)

def load_model(mn):
    path = MODEL_PATHS[mn]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, device_map="cpu",
    ).eval()
    return model, tok

try:
    rows = []
    for mn in ["TinyLlama", "Qwen"]:
        print(f"Slide 17: {mn} ...")
        model, tokenizer = load_model(mn)
        context = PROSE_CONTEXT

        bl_res = measure_perplexity(model, tokenizer, context, None, max_length=512)
        baseline_ppl = bl_res["perplexity"]
        print(f"  baseline ppl: {baseline_ppl:.2f}")

        configs = [
            ("delta_int8", "2x", "precision", CodecPress.from_codec(DeltaCodec(quantize_bits=8))),
            ("quant_int8", "2x", "precision", CodecPress.from_codec(QuantizationCodec(bits=8))),
            ("kivi_int4", "2x", "precision", CodecPress.from_kivi(bits=4, group_size=32)),
            ("svd_r0.5", "2x", "dimension", CodecPress.from_codec(SVDCodec(rank_ratio=0.5))),
            ("knorm_0.5", "2x", "eviction", KnormPress(compression_ratio=0.5)),
            ("snapkv_0.5", "2x", "eviction", SnapKVPress(compression_ratio=0.5)),
            ("delta_int4", "4x", "precision", CodecPress.from_codec(DeltaCodec(quantize_bits=4))),
            ("quant_int4", "4x", "precision", CodecPress.from_codec(QuantizationCodec(bits=4))),
            ("kivi_int2", "4x", "precision", CodecPress.from_kivi(bits=2, group_size=32)),
            ("svd_r0.25", "4x", "dimension", CodecPress.from_codec(SVDCodec(rank_ratio=0.25))),
            ("knorm_0.75", "4x", "eviction", KnormPress(compression_ratio=0.75)),
            ("snapkv_0.75", "4x", "eviction", SnapKVPress(compression_ratio=0.75)),
        ]
        for method, target, axis, press in configs:
            result = measure_perplexity(model, tokenizer, context, press, max_length=512)
            rows.append({
                "method": method, "target_reduction": target, "compression_axis": axis,
                "perplexity": result["perplexity"],
                "delta_ppl": result["perplexity"] - baseline_ppl,
                "model_name": mn, "context_type": "prose",
            })
            print(f"  {method}: ppl={result['perplexity']:.2f} delta={result['perplexity']-baseline_ppl:.2f}")
        del model

    df = pd.DataFrame(rows)
    df.to_csv(SD / "crossover_comparison.csv", index=False)
    plot_crossover_comparison(df, str(SD / "crossover_comparison.png"))
    print("SLIDE 17 DONE -> crossover_comparison.png, crossover_comparison.csv")
except Exception as e:
    traceback.print_exc()
    print(f"SLIDE 17 FAILED: {e}")
