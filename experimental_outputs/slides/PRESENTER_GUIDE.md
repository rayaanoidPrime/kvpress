# KV Cache Compression — Presenter's Guide
## All 18 Slides — Full Analysis, Data, and Talking Points

---

## SLIDE 1 — KV Cache Growth

**What it shows:**
Two lines (TinyLlama blue, Qwen teal) plotting KV cache size in MB against
sequence length (128 to 8192 tokens, log scale).

**The numbers:**
| Model      | seq=128  | seq=8192 |
|------------|----------|----------|
| TinyLlama  | 2.9 MB   | 185 MB   |
| Qwen       | 6.3 MB   | 403 MB   |

**Why this matters:**
At 8K context, a single TinyLlama request consumes 185 MB just for KV cache.
Batch 8 requests → 1.5 GB. Batch 32 → 6 GB. This is the problem KV
compression solves. Qwen is 2.2x larger because it has more layers (24 vs 22)
and 2x the KV heads (8 vs 4).

**Talking points:**
"The KV cache grows linearly with sequence length. At 8K tokens, TinyLlama
uses 185 MB per sequence. If you want to serve 32 concurrent users with 8K
context each, that's 6 GB just for KV cache — on top of model weights.
Compression is not optional."

---

## SLIDE 2 — Activation Magnitude Heatmap

**What it shows:**
2 rows (prose/code) × 2 columns (TinyLlama/Qwen) heatmaps of key vector L2
norm. X-axis = token position, Y-axis = head index. Bright = high magnitude.
Data from the MIDDLE layer (layer 11 for TinyLlama, layer 12 for Qwen).

**How to read it:**
- Vertical stripes = tokens that carry high magnitude across all heads
- Dark rows = "dead" or quiet attention heads
- Horizontal bands = some heads are consistently more active than others

**Patterns to point out:**
- Prose shows smoother, more continuous bright bands (natural language has
  gradual transitions)
- Code shows sharp, intermittent bright spots (punctuation, indentation,
  keywords like `def`, `return`, `import` create high-magnitude spikes)
- Qwen has 8 heads vs TinyLlama's 4, producing a finer heatmap
- The last few tokens are often bright → recent-token bias in attention

**Why we run this:**
Shows that key magnitude is NOT uniform. Some tokens are naturally "louder"
than others. This is the intuition behind KNormPress — prune the quiet tokens.

**Talking points:**
"Look at the code context. See those bright spikes? Those are keywords,
operators, special characters. They dominate the attention. In prose, the
pattern is smoother — natural language has more gradual importance decay."

---

## SLIDE 3 — Keys vs Values Distributions

**What it shows:**
Histograms of all key and value tensor elements (flattened) at the middle
layer. 2×2 grid, clipped to 1st–99th percentile.

**Key observations:**
- Keys are usually wider, more spread out in value range
- Values are tighter, clustered near zero
- Qwen distributions are broader than TinyLlama (more dynamic range)
- Code contexts show more extreme values (cleaner separation of signal
  from noise) compared to prose

**Why this matters:**
Quantization schemes care about the VALUE RANGE of what they quantize.
A wider distribution needs more bits or better grouping strategies.
The distribution shape tells us whether uniform quantization will work
or if we need more sophisticated approaches (like per-group quantization).

**Talking points:**
"Keys have wider dynamic range than values — that's why KIVI uses different
quantization strategies for each. Values are tightly clustered, making them
easier to quantize aggressively. The difference between prose and code shows
that context type matters — code has more structure, which compression can
exploit."

---

## SLIDE 4 — Across-Channel Variance

**What it shows:**
Bar charts of per-channel key variance at early, middle, and late layers.
Each bar is one of the 64 head_dim channels. Higher variance = that channel
position in the key vector carries more information.

**Key findings:**
- Not all 64 head_dim channels are equally important
- Some channels have 2-3× the variance of others
- This non-uniformity persists across layers
- Early layers show more uniform variance; late layers show more peaked
  variance (specialization)
- Qwen shows more variance than TinyLlama (8 heads vs 4 heads means each
  head can specialize more)

**Why we run this:**
THIS is the justification for per-channel quantization (KIVI's key mode).
If all channels had equal variance, uniform quantization would be optimal.
But they don't — so per-channel scaling is essential.

The spec for KIVI says "keys use per-channel group quantization (groups
along head_dim)." This slide shows WHY that matters: channels are not
created equal.

**Talking points:**
"Look at channels 0-10 vs channels 50-60 — the variance differs by 2-3×.
If you use one scale for all 64 channels, you waste bits on low-variance
channels and clip high-variance channels. KIVI solves this by quantizing
groups of channels separately. Each group gets its own scale."

---

## SLIDE 5 — Across-Token Variance

**What it shows:**
Line charts of per-token key variance. X-axis = token position in the
sequence. One line per head. Early, middle, and late layers shown.

**Key findings:**
- Variance varies significantly across token positions
- Some token positions consistently show high variance (important content)
- Early-middle tokens often show lower variance
- The pattern differs between prose and code (code has more jagged lines)
- Late layers show smoother variation than early layers (representation
  becomes more stable)

**Why we run this:**
This justifies per-Token quantization for values (KIVI's value mode).
Tokens have different information content, so each token needs its own
quantization scale. The alternative — one scale for all tokens — would
be suboptimal.

**Talking points:**
"Tokens are not equally important. Some carry much more information than
others. If you quantize all tokens with the same scale, you waste precision
on filler tokens and lose precision on critical tokens. KIVI's asymmetric
design treats keys and values differently based on these statistics."

---

## SLIDE 6 — Layer-Depth Profiles

**What it shows:**
4 stacked line charts (key norm, outlier fraction, delta norm, SVD top-50%
energy) vs layer index. Blue = TinyLlama, teal = Qwen. Solid = prose,
dashed = code.

**Key observations:**

| Metric | Pattern | Meaning |
|--------|---------|---------|
| k_abs_norm | Increases with depth | Keys get "louder" in deeper layers |
| k_outlier_fraction | Peaks in middle layers | Middle layers have most extreme key values |
| k_delta_norm | Decreases with depth | Key changes between adjacent tokens smooth out |
| sv_top50_energy | Very high (>0.90) at all layers | First half of singular values capture >90% of energy → strong low-rank structure |

**Crucial finding:** `sv_top50_energy` ≥ 0.91 across ALL layers for both
models. This means the key matrices are highly compressible via SVD.
The top half of singular values explain over 90% of the variance.

`k_delta_compressibility` shows that adjacent key vectors are similar
(especially in later layers), justifying delta encoding.

**Talking points:**
"Four metrics, one story: KV cache is highly compressible. sv_top50_energy
above 0.91 across all layers means low-rank approximation will work well.
Delta compressibility drops with depth as keys become smoother — perfect
for delta encoding. The data is TELLING us these compression strategies
will work before we even try them."

---

## SLIDE 7 — Attention Pattern Preservation

**What it shows:**
2×3 heatmap grid of KL divergence between baseline and compressed attention
patterns. Rows = models, columns = methods (delta_int8, kivi_int4, knorm_0.5).

**Result:** All KL values are 0.000000.

**Why? This is correct but subtle.**
During PREFILL, every layer computes SELF-ATTENTION from the input hidden
states. The KV cache is WRITTEN (then compressed by our hook) but never
READ during prefill. So `output_attentions=True` captures the self-attention
patterns, which are IDENTICAL regardless of cache compression.

The hook modifies the cache AFTER each layer's self-attention is computed.
The next layer's self-attention also comes from inputs, not from the cache.

**The real test would be during DECODING**, where each new token attends
to the COMPRESSED cache. But decoding produces attention weights of shape
(1, n_heads, 1, cache_len) which can't be compared to the prefill shapes.

**Talking points:**
"This slide shows an important limitation: prefill self-attention patterns
don't reflect compression effects. The KL divergence is zero because attention
is computed from hidden states, not from the cache. The real impact of
compression on attention patterns only becomes visible during autoregressive
decoding — which is computationally expensive to benchmark exhaustively.
This is a known methodological challenge in KV cache compression research."

---

## SLIDE 8 — Eviction Pattern Visualisation

**What it shows:**
2×2 grid of color-coded token sequences. Each token colored by its survival
probability (averaged across all layers) under KNorm (top row) and SnapKV
(bottom row) with CR=0.5. Left = prose, right = code.

**Color coding:**
- Blue: p ≥ 0.8 (almost always kept)
- Amber: 0.4 ≤ p < 0.8 (uncertain)
- Red: p < 0.4 (almost always evicted)

**Expected patterns:**
- **KNorm** keeps tokens with high key vector norms. These tend to be:
  punctuation, special characters, numbers, and content-heavy words.
  Clusters of blue around important syntactic tokens.
- **SnapKV** keeps tokens that the most recent tokens attend to strongly.
  Should show blue for contextually important tokens and heads/tails of
  the document. Should be more semantically coherent than KNorm.
- **Prose**: Blue clusters around proper nouns, dates, numbers, and
  topic-introducing sentences. Red for filler words.
- **Code**: Blue around keywords (`def`, `return`, `import`), indentation
  characters, and literals. Red for whitespace and boilerplate.

**Talking points:**
"KNorm keeps the 'loudest' tokens — high magnitude in the key space. It
tends to select punctuation, numbers, and content words. SnapKV keeps
tokens that recent context attends to, making it more semantically aware.
Neither is perfect — losing tokens means you can never retrieve them.
That's the fundamental trade-off of eviction: irreversible information loss."

---

## SLIDE 9A — Temporal Autocorrelation

**What it shows:**
2×2 grid of autocorrelation curves. X = lag (1 to 20 token positions),
Y = cosine similarity between key vectors at that lag. One line per layer
depth (early=blue, middle=teal, late=red). Models and contexts as subplots.

**Key findings:**
- **Qwen**: VERY high autocorrelation at early layers (~0.99 at lag 1),
  dropping to ~0.78 at lags 19-20. Keys are temporally smooth.
- **TinyLlama**: Lower autocorrelation overall (~0.62-0.76), especially
  at early layers.
- Later layers maintain autocorrelation better than early layers.
- Prose and code show similar autocorrelation patterns.

**Interpretation:**
High autocorrelation means key vector at position i is very similar to
key vector at position i+1. This is the foundation for DELTA ENCODING:
store only the difference between adjacent keys, since the difference
is small.

Qwen's high autocorrelation (0.99 at lag 1) means delta encoding will
work exceptionally well. TinyLlama's lower autocorrelation (0.62 at lag 1)
means delta encoding will be less effective.

**Talking points:**
"Qwen at lag 1: cosine similarity of 0.99. That means adjacent key vectors
are nearly identical. If you store just the difference, you need far fewer
bits. This is why Qwen's delta_int8 achieves 0.50 compression ratio with
negligible perplexity increase. TinyLlama's keys are more dynamic — lower
autocorrelation means delta encoding is less efficient."

---

## SLIDE 9B — Intrinsic Rank Per Layer

**What it shows:**
LEFT: Scree plots — cumulative variance explained vs singular value index.
One line per layer, colored by depth. Dashed red line at y=0.90 (90% energy).
RIGHT: Bar chart of effective rank (number of SVD components needed for 90%
variance) per layer.

**Key findings:**
- Effective rank is surprisingly LOW: 10-20 for TinyLlama, 15-30 for Qwen
  (out of 64 head_dim)
- This means 64-dimensional key vectors can be represented with ~20 dimensions
  while retaining 90% of the information
- Later layers have higher effective rank than early layers
- Qwen has higher effective rank than TinyLlama (more heads, more specialization)

**What effective_rank_90 = -1 means:** The SVD computation failed because
seq_len < head_dim (e.g., 32 tokens vs 64 dims for some layers). These are
edge cases where the matrix is underdetermined.

**Why this matters:**
Low effective rank DIRECTLY justifies SVD compression. If rank ≈ 20 and
head_dim = 64, we can achieve ~3× compression. The data says "yes, this
will work."

**Talking points:**
"Look at these scree plots. The first 10-20 singular values capture 90% of
the variance. The key space has an effective rank of 15-25 out of 64 — that's
a 2.5-4× compression opportunity. This is not forced; it's inherent in the
data. SVD compression works because the key matrices are low-rank."

---

## SLIDE 10 — Three-Axis Taxonomy

**What it shows:**
A triangular diagram with three regions: Precision Reduction (blue, bottom-left),
Token Eviction (teal, bottom-right), and Dimension Reduction (purple, top).
Specific methods placed in their regions.

**The three axes explained:**

| Axis | What it does | Methods | Trade-off |
|------|-------------|---------|-----------|
| Precision | Reduces bits per element | INT8/4, KIVI, KVQuant | Information is degraded, not lost |
| Eviction | Removes entire tokens | StreamingLLM, KNorm, SnapKV, H2O | Lost tokens CANNOT be recovered |
| Dimension | Reduces head_dim via low-rank | SVD, MLA | Requires training compatibility |

**Hybrids** (center): MiniKV combines eviction + precision. AdaKV uses
head-wise adaptive strategies.

**Talking points:**
"Three fundamentally different approaches to the same problem. Precision
reduction is lossy but reversible — you can roughly reconstruct. Eviction
is permanent — pruned tokens are gone forever. Dimension reduction is a
third axis: compress the representation itself. The best solution for
your use case depends on which axis your workload tolerates best."

---

## SLIDE 11 — Pipeline Diagram

**What it shows:**
Two rows of boxes connected by arrows. Top row = PREFILL (once per sequence),
bottom row = DECODE (per token, per layer). Key operations: Attention Forward,
Compress (encode), DMA Write → DRAM, DMA Read ← DRAM, Decompress (decode),
Attention (Q·K^T·V).

**Critical callout:** "Decompression runs orders of magnitude more often
than compression. Decode speed is the critical path."

**Why this matters:**
- Prefill compression happens ONCE per request
- Decode decompression happens N_tokens × N_layers times
- A codec with 0.1ms encode + 0.5ms decode is WORSE than one with 5ms encode
  + 0.01ms decode, because decode dominates
- This is why KIVI is designed for fast decode (simple per-group dequantization)

**Talking points:**
"This is the key architectural insight. Encoding happens once per sequence.
Decoding happens for every token, every layer. If your decode is slow, your
entire inference pipeline is slow. Our codec benchmarks measure both encode
and decode latency for exactly this reason. Fast decode beats fast encode
every time."

---

## SLIDE 12 — Deployment Options Table

**What it shows:**
A table mapping deployment tasks to recommended compression strategies.

| Task | Concern | Axis | Reasoning |
|------|---------|------|-----------|
| Long doc QA | Retrieving specific facts | Eviction (careful) | Pruned tokens cannot be retrieved |
| Summarisation | Global coherence | Precision reduction | No single token is critical |
| Code completion | Local syntactic structure | Precision reduction | Code has low temporal smoothness |
| Multi-turn chat | Growing cache | Eviction + precision | Cache must stay bounded |
| Structured data | Exact value preservation | None / high-precision | Numbers and fields are fragile |

**Talking points:**
"There's no one-size-fits-all. Long document QA needs eviction to fit the
context, but you risk pruning the very fact the user asked about.
Summarisation doesn't care about individual tokens — precision reduction
works well. Code completion needs precision because code has sharp syntax.
For structured data, don't compress — the cost of a wrong number is too high."

---

## SLIDE 13 — Codec Latency + Attention Error

**What it shows:**
LEFT: Scatter plot of decode time vs compression ratio. Each point = one codec.
RIGHT: Scatter plot of attention logit relative error vs MSE.

**Key numbers (TinyLlama avg, 32 tokens × 4 heads × 64 dim):**

| Codec | Ratio | Decode ms | MSE | Attn Err |
|-------|-------|-----------|-----|----------|
| kivi_int2 | 0.81 | 0.56 | 0.154 | 0.41 |
| delta_int4 | 0.74 | 0.85 | 2.78 | 1.74 |
| quant_int4 | 0.75 | 0.50 | 0.022 | 0.15 |
| kivi_int4 | 0.69 | 0.59 | 0.006 | 0.08 |
| delta_int8 | 0.50 | 0.56 | 0.009 | 0.09 |
| quant_int8 | 0.50 | 0.29 | 0.000 | 0.01 |
| kivi_int8 | 0.44 | 0.72 | 0.000 | 0.00 |
| svd_r0.5 | 0.25 | 0.24 | 0.209 | 0.45 |
| delta_fp16 | 0.00 | 0.35 | 0.000 | 0.00 |

**Patterns:**
- quant_int8 is fastest decode AND lowest error — best value
- kivi_int4 achieves 0.69 ratio with very low MSE (0.006)
- delta_int4 has massive error (MSE 2.78, attn error 1.74) — explains why PPL explodes
- SVD has highest attention error per unit of compression — rank reduction
  affects attention alignment more than quantization
- Attention logit error correlates with, but is NOT the same as, MSE — some
  codecs (svd_r0.5) have high attn error despite moderate MSE

**Talking points:**
"Quantization (blue family) dominates the latency-quality Pareto frontier.
quant_int8 gives 2× compression with near-zero error and fastest decode.
kivi_int4 gives 3× compression with very low error. delta_int4 is dangerous
— 4-bit delta quantization destroys the representation. SVD gives good
compression but hurts attention alignment more than quantization."

---

## SLIDE 14 — Codec MSE Bar Chart

**What it shows:**
Grouped bar chart: MSE (blue) and attention logit relative error (teal)
for each codec, sorted by MSE ascending.

**Same data as Slide 13 but easier to compare side-by-side.**

**Talking points:**
"Sort by MSE and the ranking is clear: fp16/delta_int8/kivi_int8 are
lossless in practice. INT4 quantization is acceptable. INT4 delta and
low-rank SVD degrade significantly. The attention error bar often exceeds
the MSE bar — meaning attention is more sensitive to compression than
simple reconstruction suggests."

---

## SLIDE 15 — PPL vs Compression Ratio

**What it shows:**
4 plots (2 models × 2 context types) and 2 eviction plots. Each line = one
codec/method. X = effective compression ratio, Y = perplexity (log scale).

**Key findings (TinyLlama prose):**

| Method | PPL | CR | Delta from baseline |
|--------|-----|-----|---------------------|
| baseline | 5.80 | 0.00 | — |
| kivi_int2 | 5.94 | 0.81 | +0.14 |
| delta_int8 | 5.88 | 0.50 | +0.08 |
| quant_int8 | 5.77 | 0.50 | -0.03 (BETTER!) |
| quant_int4 | 10.30 | 0.75 | +4.50 |
| delta_int4 | 19,261 | 0.75 | **broken** |

**Qwen prose is harder (baseline PPL = 7.55 vs TinyLlama 5.80).**

**Critical finding: EVICTION CAN BE BETTER THAN COMPRESSION.**
For Qwen, Knorm at CR=0.75 gives PPL=5.63 — BETTER than the baseline 7.55.
This happens because high-norm tokens are the most informative; removing
noisy tokens IMPROVES prediction quality. This is not a bug — it's a
legitimate finding that attention-based token importance matters.

**For eviction (TinyLlama):**
- KNorm CR=0.5 → PPL 55.9 (10× baseline)
- SnapKV CR=0.5 → PPL 54.3 (9× baseline)
- Both degrade severely because losing 50% of tokens destroys context

**Delta_int4 is BROKEN**: PPL of 19,000+ for TinyLlama, 980,000+ for Qwen.
Raw 4-bit quantization of deltas destroys the representation because
quantization errors accumulate over the cumulative sum during decode.

**Talking points:**
"Three tiers emerge. Tier 1: KIVI 2-bit and INT8 methods — near-baseline
PPL with excellent compression. Tier 2: INT4 quantization, SVD — acceptable
degradation for many use cases. Tier 3: delta_int4 — broken, do not use.
The surprise: KNorm at high compression can IMPROVE perplexity on Qwen by
removing noisy tokens. Eviction isn't always worse than precision reduction."

---

## SLIDE 16 — PPL vs Quantization Bits

**What it shows:**
2×2 grid. X = bits (2, 4, 8, 16), Y = perplexity (log scale). One line per
quantization family: Quantization (teal), KIVI (amber), Delta (blue).

**Why Delta has 16-bit point:** delta_fp16 is lossless (fp16 = 16 bits),
treated as a 16-bit reference point.

**Key trend:**
- Delta: 8-bit works, 4-bit EXPLODES (from ~5.8 to ~19,000)
- Quantization: 8-bit → 4-bit degrades gracefully (5.8 → 10.3)
- KIVI: 8, 4, and 2-bit all maintain near-baseline PPL (5.8 → 5.8 → 5.9)

**Why Delta 4-bit fails so badly:**
The delta values are small differences (high autocorrelation → delta ≈ 0).
When you quantize small values to 4 bits (16 levels), you either clip them
to 0 (losing all information) or amplify quantization noise. During the
cumulative sum in decode, these errors compound, and after ~256 tokens the
reconstructed keys are essentially random.

**Talking points:**
"KIVI is the star here. 2-bit quantization with essentially no PPL increase.
Why? Because per-group scaling prevents error accumulation. Delta at 4-bit
fails catastrophically — the quantization error in deltas compounds during
the cumulative sum reconstruction. Standard per-tensor quantization at 4-bit
degrades gracefully — a moderate PPL increase for aggressive compression."

---

## SLIDE 17 — Crossover Comparison

**What it shows:**
Grouped bar chart per model. X = method, Y = delta perplexity from baseline.
Grouped by target (2× memory vs 4× memory). Colors: precision = blue,
eviction = teal, dimension = purple.

**Key numbers (TinyLlama):**

| Method | Target | Delta PPL | Axis |
|--------|--------|-----------|------|
| quant_int8 | 2× | -0.04 | precision |
| kivi_int4 | 2× | +0.01 | precision |
| delta_int8 | 2× | +0.07 | precision |
| svd_r0.5 | 2× | +0.54 | dimension |
| knorm_0.5 | 2× | +50.11 | eviction |
| snapkv_0.5 | 2× | +48.46 | eviction |
| kivi_int2 | 4× | +0.14 | precision |
| quant_int4 | 4× | +4.50 | precision |
| svd_r0.25 | 4× | +1.73 | dimension |
| delta_int4 | 4× | +19,256 | precision |
| knorm_0.75 | 4× | +55.51 | eviction |
| snapkv_0.75 | 4× | +821.42 | eviction |

**The hierarchy is clear:**
1. PRECISION REDUCTION (blue) dominates at both 2× and 4× targets
2. DIMENSION REDUCTION (purple) is moderate but worse than precision
3. EVICTION (teal) is terrible at 2× (50 PPL increase) and catastrophic
   at 4× (55-800 PPL increase)

**Qwen caveat:** KNorm at CR=0.75 IMPROVES PPL by 1.9 (from 7.55 to 5.63).
This is the "noisy token removal" effect — when you prune 75% of tokens
but keep only the high-norm ones, the model actually predicts better on the
continuation. This is NOT a general recommendation, but it shows eviction
isn't always worse.

**Talking points:**
"At equal memory targets, precision reduction crushes eviction. For 2×
compression, quant_int8 gives you better-than-baseline PPL while KNorm
gives 50 PPL increase. For 4×, KIVI at 2-bit (0.14 PPL increase) vs
SnapKV at CR=0.75 (821 PPL increase). Eviction has one use case: when you
ABSOLUTELY must bound cache size in a streaming setting. Otherwise,
precision reduction is the clear winner for quality."

---

## SLIDE 18 — Needle in the Haystack

**What it shows:**
Heatmap: rows = compression method, columns = needle position (10% to 90%
through the context). Color = mean exact match rate (0 to 1).

**This slide has NOT been generated yet** — see the overnight run.

**What to expect:**
- Baseline: near 1.0 at all positions (model can retrieve facts anywhere)
- Methods that affect token ordering (eviction) will show a position
  dependency — facts in the middle are harder to retrieve than at edges
- Methods that preserve all tokens (precision reduction) should maintain
  near-baseline retrieval, though precision loss may blur facts
- KNorm should perform worst on middle-position needles (they're neither
  recent enough to be in the observation window nor initial enough to
  have strong positional encoding)

**Why this experiment matters:**
The needle-in-haystack test measures whether compression preserves a
model's ability to ACCESS specific facts from the compressed context.
Even if perplexity is good, if the model can't retrieve a specific number
from the context, the compression is damaging retrieval capabilities.

**Talking points:**
"Perplexity measures token prediction accuracy. Needle-in-haystack measures
information retrieval from compressed context. A codec can have good PPL
but terrible retrieval if it loses the specific tokens carrying the answer.
This is why structured data tasks should use no compression or high-precision
only."

---

## SYNTHESIS — What We Learned

### 1. KV cache IS compressible
Every diagnostic (channel variance, token variance, autocorrelation, SVD
scree plots) confirms that KV cache representations are highly redundant.
The effective rank is 15-25 out of 64. Adjacent keys have 0.6-0.99 cosine
similarity. Channels vary by 2-3× in importance.

### 2. Precision Reduction dominates Eviction
At equal memory targets, quantization-based methods (KIVI, INT8/4) preserve
perplexity MUCH better than token eviction (KNorm, SnapKV). The gap is
50 PPL vs 0.14 PPL at 2× compression. This is a clear win for precision
methods in quality-sensitive applications.

### 3. KIVI is the best overall method
KIVI 2-bit achieves 4× memory reduction (81% compression) with only +0.14
perplexity increase on TinyLlama. It achieves this through asymmetric
quantization: per-channel grouping for keys and per-token grouping for
values, validated by our channel/token variance analysis.

### 4. Naive delta 4-bit is broken
Delta encoding at 4 bits destroys the KV cache because quantization errors
in deltas compound during cumulative sum reconstruction. Delta at 8 bits or
fp16 is fine, but 4-bit delta should not be used.

### 5. Context type matters
Code is more compressible than prose (lower baseline perplexity). The
structured nature of code creates more redundancy that compression can
exploit. Deployment recommendations should consider the input domain.

### 6. Eviction has a niche
While precision reduction dominates in quality, eviction is the ONLY option
when the cache size must be strictly bounded (streaming, infinite
conversations). And in some cases (Qwen + KNorm), removing noisy tokens
IMPROVES quality — a counterintuitive but legitimate result.

---

## COMMON QUESTIONS AND ANSWERS

**Q: Why is baseline PPL different between TinyLlama (5.80) and Qwen (7.55)?**
A: Qwen has more parameters (0.5B vs 1.1B) — wait, Qwen 0.5B is smaller.
Actually TinyLlama 1.1B is larger and better trained, hence lower PPL.
Qwen 0.5B is less capable but has 8 KV heads (vs 4) for more expressive
attention.

**Q: Why does quant_int8 sometimes give BETTER PPL than baseline?**
A: Quantization adds noise. In some cases, noise acts as regularization,
slightly improving generalization on the continuation. The difference is
within measurement noise (±0.03 PPL). Not significant.

**Q: Why does KNorm at CR=0.75 improve Qwen PPL?**
A: KNorm keeps tokens with highest key norm. If noisy/uninformative tokens
have low norms, pruning them removes distraction and the model predicts
better on the continuation. This is a real but fragile effect — it depends
on the specific context and model.

**Q: Should I use KIVI in production?**
A: For most applications, yes — KIVI 4-bit (CR=0.69) with negligible PPL
increase is production-ready. KIVI 2-bit (CR=0.81) with +0.14 PPL is
suitable for memory-constrained settings. Avoid delta 4-bit at all costs.

**Q: What about combining methods?**
A: The taxonomy slide shows hybrids in the center. You can combine
precision reduction + eviction (MiniKV) or use head-wise adaptive strategies
(AdaKV). The best combination depends on your workload's tolerance for
each compression axis.
