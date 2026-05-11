"""SLIDE 18 - Needle in the Haystack (needs model, TinyLlama only)."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import pandas as pd
from experimental.harness import MODEL_PATHS, needle_in_haystack
from experimental.context_samples import PROSE_CONTEXT, NEEDLE_FACTS
from experimental.plots import plot_needle_heatmap
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress.codecs import DeltaCodec, KIVICodec
from kvpress.presses.codec_press import CodecPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress

SD = Path("experimental_outputs/slides/slide18")
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
    model_name = "TinyLlama"
    print(f"Slide 18: {model_name} ...")
    model, tokenizer = load_model(model_name)

    press_grid = {
        "baseline": None,
        "knorm_0.5": KnormPress(compression_ratio=0.5),
        "snapkv_0.5": SnapKVPress(compression_ratio=0.5),
        "delta_int8": CodecPress.from_codec(DeltaCodec(quantize_bits=8)),
        "kivi_int4": CodecPress.from_kivi(bits=4, group_size=32),
    }

    positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    runs_per_cell = 3

    raw_rows = []
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
                    max_context_tokens=512,
                    press=press,
                )
                raw_rows.append({
                    "press_name": press_name, "needle_position": pos,
                    "run_idx": run_idx, "needle_fact": needle_fact,
                    "exact_match": result["exact_match"], "f1": result["f1"],
                    "answer": result["answer"], "context_tokens": result["context_tokens"],
                    "inference_ms": result["inference_ms"],
                })
                print(f"  {press_name} pos={pos} run={run_idx}: em={result['exact_match']} f1={result['f1']:.2f}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(SD / "needle_raw.csv", index=False)

    agg = raw_df.groupby(["press_name", "needle_position"]).agg(
        mean_exact_match=("exact_match", "mean"),
        mean_f1=("f1", "mean"),
    ).reset_index()
    agg.to_csv(SD / "needle_aggregated.csv", index=False)

    plot_needle_heatmap(agg, str(SD / "needle_heatmap.png"))
    print("SLIDE 18 DONE -> needle_heatmap.png, needle_raw.csv, needle_aggregated.csv")
except Exception as e:
    traceback.print_exc()
    print(f"SLIDE 18 FAILED: {e}")
