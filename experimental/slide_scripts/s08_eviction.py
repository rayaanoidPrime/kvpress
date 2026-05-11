"""SLIDE 8 - Eviction Pattern Visualisation (needs model)."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from experimental.harness import MODEL_PATHS, MODEL_ARCH, tokenize
from experimental.context_samples import PROSE_CONTEXT, CODE_CONTEXT
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.snapkv_press import SnapKVPress

SD = Path("experimental_outputs/slides/slide08")
SD.mkdir(exist_ok=True)

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
    CH = "#2563EB"
    CM = "#D97706"
    CL = "#DC2626"

    for model_name in ["TinyLlama", "Qwen"]:
        print(f"Slide 8: {model_name} ...")
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

        fig, axes = plt.subplots(2, 2, figsize=(20, max(8, 4 * 2)), facecolor="white")
        methods = ["KNorm (CR=0.5)", "SnapKV (CR=0.5)"]
        ctx_labels = ["prose", "code"]
        for ri in range(2):
            for ci in range(2):
                ax = axes[ri, ci]
                toks, prbs = results[ri][ci]
                nt = len(toks)
                pr = min(100, nt)
                nr = max(1, int(np.ceil(nt / pr)))
                for i, (tk, pb) in enumerate(zip(toks, prbs)):
                    rw = i // pr
                    cc = i % pr
                    if pb >= 0.8:
                        col = CH
                    elif pb >= 0.4:
                        col = CM
                    else:
                        col = CL
                    ax.text(cc, nr - rw - 1, tk, fontfamily="monospace", fontsize=4.5,
                            color=col, ha="center", va="center")
                ax.set_xlim(-0.5, pr - 0.5)
                ax.set_ylim(-0.5, nr - 0.5)
                ax.set_title(f"{methods[ri]} x {ctx_labels[ci]}")
                ax.axis("off")
        fig.suptitle(f"Eviction Pattern - {model_name}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(str(SD / f"eviction_pattern_{model_name.lower()}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        del model
        print(f"  {model_name} done")
    print("SLIDE 08 DONE -> eviction_pattern_tinyllama.png, eviction_pattern_qwen.png")
except Exception as e:
    traceback.print_exc()
    print(f"SLIDE 08 FAILED: {e}")
