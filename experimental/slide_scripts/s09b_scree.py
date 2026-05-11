"""SLIDE 9B - Scree plot and Effective Rank (uses instrumented data)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import numpy as np
import pandas as pd
from experimental.plots import plot_scree, plot_effective_rank

SD = Path("experimental_outputs/slides/slide09b")
SD.mkdir(exist_ok=True)
INST = Path("experimental_outputs/slides/instrumented")

try:
    sc_dict = {}
    er_rows = []
    for mn in ["TinyLlama", "Qwen"]:
        sc_dict[mn] = {}
        for ctx in ["prose", "code"]:
            d = INST / mn / ctx
            st = json.loads((d / "instrumented_stats.json").read_text())
            st = {int(k): v for k, v in st.items()}
            for li in sorted(st.keys()):
                sc_dict[mn][li] = st[li].get("sv_cumvar", [])
                er_rows.append({"model_name": mn, "context_type": ctx, "layer_idx": li,
                               "effective_rank_90": st[li].get("effective_rank_90", -1)})
    er_df = pd.DataFrame(er_rows)
    er_df.to_csv(SD / "effective_rank.csv", index=False)
    plot_scree(sc_dict, str(SD / "scree_plot.png"))
    plot_effective_rank(er_df, str(SD / "effective_rank_bar.png"))
    print("SLIDE 09B DONE -> scree_plot.png, effective_rank_bar.png, effective_rank.csv")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"SLIDE 09B FAILED: {e}")
