"""SLIDE 16 - PPL vs Quantization Bits (derived from slide 15 data)."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
from experimental.plots import plot_ppl_vs_bits

SD = Path("experimental_outputs/slides/slide16")
SD.mkdir(exist_ok=True)

SRC = Path("experimental_outputs/slides/slide15/ppl_codec_sweep_all.csv")

try:
    df = pd.read_csv(SRC)
    codec_to_family_bits = {
        "quant_int8": ("Quantization", 8),
        "quant_int4": ("Quantization", 4),
        "kivi_int8": ("KIVI", 8),
        "kivi_int4": ("KIVI", 4),
        "kivi_int2": ("KIVI", 2),
        "delta_int8": ("Delta", 8),
        "delta_int4": ("Delta", 4),
        "delta_fp16": ("Delta", 16),
    }

    rows = []
    for _, r in df.iterrows():
        cn = r["codec_name"]
        if cn in codec_to_family_bits:
            family, bits = codec_to_family_bits[cn]
            rows.append({
                "family": family, "bits": bits, "perplexity": r["perplexity"],
                "model_name": r["model_name"], "context_type": r["context_type"],
                "codec_name": cn,
            })

    bits_df = pd.DataFrame(rows)
    bits_df.to_csv(SD / "ppl_vs_bits.csv", index=False)
    plot_ppl_vs_bits(bits_df, str(SD / "ppl_vs_bits.png"))
    print("SLIDE 16 DONE -> ppl_vs_bits.png, ppl_vs_bits.csv")
except Exception as e:
    traceback.print_exc()
    print(f"SLIDE 16 FAILED: {e}")
