"""SLIDE 12 - Deployment Options Table (static)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SD = Path("experimental_outputs/slides/slide12")
SD.mkdir(exist_ok=True)

try:
    data = [
        ["Long doc QA", "Retrieval of\nspecific facts", "Eviction\n(careful)", "Pruned tokens\ncannot be retrieved"],
        ["Summarisation", "Global coherence", "Precision\nreduction", "No single token\nis critical"],
        ["Code completion", "Local syntactic\nstructure", "Precision\nreduction", "Code has low\ntemporal smoothness"],
        ["Multi-turn chat", "Growing cache", "Eviction +\nprecision", "Cache must\nstay bounded"],
        ["Structured data", "Exact value\npreservation", "None or\nhigh-precision", "Numbers and fields\nare fragile"],
    ]
    cols = ["Task", "Memory Concern", "Suggested Axis", "Reasoning"]

    fig, ax = plt.subplots(figsize=(14, 3.8), facecolor="white")
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=cols, cellLoc="center", loc="center",
                   colWidths=[0.16, 0.24, 0.18, 0.42])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1B2A4A")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F0F4FA" if row % 2 == 1 else "white")
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)
    ax.set_title("Deployment Options: Choosing a KV Cache Compression Strategy",
                 fontsize=12, fontweight="bold", color="#1B2A4A", pad=18)
    fig.tight_layout()
    fig.savefig(str(SD / "deployment_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("SLIDE 12 DONE -> deployment_table.png")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"SLIDE 12 FAILED: {e}")
