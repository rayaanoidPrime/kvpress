"""SLIDE 11 - Pipeline diagram (static, no model)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

SD = Path("experimental_outputs/slides/slide11")
SD.mkdir(exist_ok=True)

try:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)

    def bx(ax, x, y, w, h, text, color="#1B2A4A"):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     facecolor=color, edgecolor="white", alpha=0.9)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    def ar(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2563EB", lw=2))

    ax.text(0.5, 4.0, "PREFILL\n(once per\nsequence)", ha="center", fontsize=9, fontweight="bold", color="#1B2A4A")
    bx(ax, 2.0, 3.2, 2.8, 1.2, "Attention\nForward")
    ar(ax, 4.8, 3.8, 5.5, 3.8)
    bx(ax, 5.5, 3.2, 2.0, 1.2, "Compress\n(encode)")
    ar(ax, 7.5, 3.8, 8.2, 3.8)
    bx(ax, 8.2, 3.2, 2.8, 1.2, "DMA Write\n-> DRAM")

    ax.text(0.5, 1.5, "DECODE\n(per token\nper layer)", ha="center", fontsize=9, fontweight="bold", color="#1B2A4A")
    bx(ax, 2.0, 0.8, 2.8, 1.2, "DMA Read\n<- DRAM")
    ar(ax, 4.8, 1.4, 5.4, 1.4)
    bx(ax, 5.4, 0.8, 2.4, 1.2, "Decompress\n(decode)")
    ar(ax, 7.8, 1.4, 8.4, 1.4)
    bx(ax, 8.4, 0.8, 2.8, 1.2, "Attention\n(Q.K^T.V)")
    ar(ax, 11.2, 1.4, 11.8, 1.4)
    bx(ax, 11.8, 0.8, 1.8, 1.2, "DMA Write\n-> DRAM")

    c = mpatches.FancyBboxPatch((2.5, 0.1), 9.5, 0.5, boxstyle="round,pad=0.1",
                                 facecolor="#D97706", edgecolor="white", alpha=0.12)
    ax.add_patch(c)
    ax.text(7.25, 0.35, "Decompression runs orders of magnitude more often than compression. Decode speed is the critical path.",
            ha="center", fontsize=9, color="#D97706", fontstyle="italic")
    ax.set_title("KV Cache Compression Pipeline", fontsize=14, fontweight="bold", color="#1B2A4A")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(SD / "pipeline_diagram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("SLIDE 11 DONE -> pipeline_diagram.png")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"SLIDE 11 FAILED: {e}")
