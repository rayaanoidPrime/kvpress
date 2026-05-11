"""SLIDE 9A - Temporal Autocorrelation (uses instrumented data from disk)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import numpy as np
import pandas as pd
from experimental.plots import plot_autocorrelation
from experimental.harness import MODEL_ARCH

SD = Path("experimental_outputs/slides/slide09a")
SD.mkdir(exist_ok=True)
INST = Path("experimental_outputs/slides/instrumented")

try:
    ac_dict = {}
    rows = []
    for mn in ["TinyLlama", "Qwen"]:
        ac_dict[mn] = {}
        for ctx in ["prose", "code"]:
            d = INST / mn / ctx
            st = json.loads((d / "instrumented_stats.json").read_text())
            st = {int(k): v for k, v in st.items()}
            layers = sorted(st.keys())
            e, m, l = layers[0], layers[len(layers) // 2], layers[-1]
            ac_dict[mn][ctx] = {
                "early": st[e]["autocorr_lags_1_to_20"],
                "middle": st[m]["autocorr_lags_1_to_20"],
                "late": st[l]["autocorr_lags_1_to_20"],
            }
            for dp, li in [("early", e), ("middle", m), ("late", l)]:
                for i, val in enumerate(st[li]["autocorr_lags_1_to_20"]):
                    rows.append({"model": mn, "context": ctx, "depth": dp, "layer": li, "lag": i+1, "cosine_sim": val})
    pd.DataFrame(rows).to_csv(SD / "autocorrelation.csv", index=False)
    plot_autocorrelation(ac_dict, str(SD / "autocorrelation.png"))
    print("SLIDE 09A DONE -> autocorrelation.png, autocorrelation.csv")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"SLIDE 09A FAILED: {e}")
