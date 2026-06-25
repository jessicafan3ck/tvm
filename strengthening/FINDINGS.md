# TVM — Strengthening the empirical core

Re-analysis of the TVM datasets to test the paper's central claim rigorously.
The paper concluded *"aesthetic sentiment is not visually encoded — it cannot be
recovered from objective features,"* based on unsupervised k-means failing to
recover the labels. That test is too weak. Below we test it the right way
(supervised probes, confound controls, a semantic ceiling) and the conclusion
flips.

## TL;DR — the paper's headline claim is wrong as stated

Vibe **is** partially and robustly encoded in objective composition. It is just
**distributed, not cluster-separable**, which is why unsupervised geometry missed
it. Semantic vision (CLIP) recovers ~2× more, and the two are complementary —
empirically motivating the multimodal `F_obj + F_sem + F_comp` architecture.

## Datasets
- **Curated**: 4,994 images, 9 aesthetic modifiers (per-image F_obj JSONs).
  Heavily imbalanced (romantic 1048 … whimsical 131).
- **Benchmark**: 812 images, 8 emotions (EmotionROI/FI subset). F_obj JSONs +
  source images both available and matched 1:1 (812/812 resolved).

## Results

### 1. Supervised probe — is vibe recoverable from F_obj? (`probe.py`)
GradientBoosting, 5-fold stratified. Permutation test = true chance floor.

| Dataset | balanced acc | macro-F1 | true chance (permuted) | ratio |
|---|---|---|---|---|
| Curated (9 modifiers) | **0.451** | 0.463 | 0.113 | ~4.0× |
| Benchmark (8 emotions) | **0.335** | 0.333 | 0.126 | ~2.7× |

→ Vibe is clearly recoverable, far above chance, on **both** datasets.
This directly contradicts "not visually encoded."

### 2. Subject confound — is it just scene/content leakage? (`probe.py`)
- F_obj → subject (26 classes): balanced acc 0.278 (encodes scene too).
- **F_obj → modifier with GroupKFold by subject** (subjects can't leak):
  0.451 → **0.421**. Barely drops → the vibe signal is **subject-independent**, not
  a content artifact.

### 3. Where the signal lives (`probe.py`, ANOVA + mutual information)
Top discriminative features are **color / tonal / visual-mass**, NOT the
framing/subject-isolation axis the paper highlighted:
`mean_visual_weight` (F=417), `warm_color_ratio` (322), `shadow_ratio` (184),
`midtone_ratio`, `mean_contrast`, `cool_color_ratio`.
The paper's "subject isolation 12.4×" drove the *outlier* split (Test 5), not the
*modifier* signal.

### 4. The "0.9943 silhouette / universal k=2" is a scaling artifact (`probe.py`)
Three features are unbounded after RobustScaler (max |z| = 62038, 4855, 267).
A few extreme values dominate Euclidean distance → k-means peels off 30 outliers
→ silhouette ≈ 0.99.
- RobustScaler only (paper): silhouette(k=2) = **0.994**
- Winsorize 1–99% + StandardScaler (correct): silhouette(k=2) = **0.119**
The cross-dataset "universality" was just both datasets sharing this pathology.
NMI(clusters vs labels) ≈ 0.00–0.07 in all preprocessings → modifiers genuinely
are NOT cluster-separable (the paper's *observation* is right; its *inference*
"therefore absent" is wrong).

### 5. Floor vs ceiling — CLIP image embeddings on the SAME images
CLIP ViT-B/32. `clip_ceiling.py` (benchmark), `clip_ceiling_curated.py` (curated).

**Benchmark — 812 images, 8 emotions (chance 0.125):**
| Representation | balanced acc | macro-F1 | vs chance |
|---|---|---|---|
| F_obj | 0.335 | 0.333 | 2.7× |
| CLIP-image (linear) | 0.617 | 0.619 | 4.9× |
| CLIP-image (GBM) | 0.619 | 0.633 | 5.0× |
| **F_obj + CLIP (linear)** | **0.640** | 0.641 | 5.1× |

**Curated — 4,994 images, 9 modifiers (chance 0.111):**
| Representation | balanced acc | macro-F1 | vs chance |
|---|---|---|---|
| F_obj | 0.451 | 0.463 | 4.1× |
| CLIP-image (linear) | 0.687 | 0.686 | 6.2× |
| CLIP-image (GBM) | 0.708 | 0.722 | 6.4× |
| **F_obj + CLIP (linear)** | **0.699** | 0.698 | 6.3× |
| CLIP, GroupKFold by subject | 0.581 | 0.582 | 5.2× |

→ Both datasets: semantic vision carries ~2× the vibe signal of composition.
**F_obj adds complementary signal on top of CLIP under a matched classifier**
(benchmark 0.617→0.640; curated 0.687→0.699) — it is not redundant.
CLIP signal survives subject-grouping (0.687→0.581) — mostly real vibe, ~10pt
subject leakage.

### 6. Confusion is semantic overlap, not failure (`clip_ceiling_curated.py`)
Top CLIP confusions are near-synonymous modifiers:
`introspective→melancholic 0.35`, `haunted→melancholic 0.21`,
`whimsical→ethereal 0.14`, `candid→romantic 0.14`.
The "errors" largely reflect genuine overlap in the modifier taxonomy — direct
evidence that vibe is *distributed/overlapping*, not cleanly separable, and that
some label pairs are not perceptually distinct.

### 7. Raising the ceiling — scale saturates; residual is subjectivity
(`ceiling_backbones.py`, `ceiling_openclip.py`, `collapse_test.py`)

Curated 9-modifier ceiling vs backbone (linear probe, chance 0.111):
| Backbone | balanced acc | macro-F1 |
|---|---|---|
| F_obj (floor) | 0.448 | — |
| CLIP ViT-B/32 | 0.687 | 0.686 |
| CLIP ViT-B/16 | 0.687 | 0.689 |
| CLIP ViT-L/14 | 0.707 | 0.708 |
| SigLIP-SO400M-384 | **0.731** | 0.729 |
| F_obj + SigLIP | 0.734 | — |

→ The ceiling rises with model scale but **saturates** (+0.044 B→SigLIP, shrinking).
F_obj is complementary only to *weak* backbones (adds +0.012 on CLIP-B/32, ~0 on SigLIP).

**Per-modifier P/R, SigLIP-SO400M** (hardest = introspective):
candid .766 · ethereal .667 · haunted .733 · **introspective .451** · melancholic .731 ·
nostalgic .749 · radiant .774 · romantic .817 · whimsical .873.

**Capacity vs subjectivity (progressive merge of most-confused pair, re-probe):**
| #classes | raw acc | merged |
|---|---|---|
| 9 | 0.733 | — |
| 8 | 0.784 | introspective+melancholic |
| 7 | 0.813 | +haunted |
| 6 | 0.819 | ethereal+radiant |
| 5 | 0.858 | +romantic |
| 4 | 0.875 | +nostalgic |

→ One merge (introspective↔melancholic, the #1 confused pair, near-synonyms) recovers
**+5 pts**. A large share of residual error is **taxonomy overlap / label subjectivity**,
not model capacity. introspective is not perceptually distinct from melancholic.

**Generalizes to the benchmark** (`collapse_bench.py`, CLIP-B/32, 8 emotions, 0.622
raw acc): merging the most-confused pair recovers the same way and follows the
valence–arousal circumplex — 8→7 fear+sad 0.676; →5 amusement+excitement 0.756;
→3 awe+contentment 0.829. Same conclusion on an independent dataset.
Composition handles color/tonal vibes (radiant, melancholic); semantic vision is needed
for relational vibes (candid, whimsical, romantic); the subjective one (introspective)
resists every backbone.

## Corrected thesis for the paper
> Aesthetic vibe is robustly but partially encoded in objective composition
> (≈3–4× chance, subject-independent, generalizing across two datasets), carried
> mainly by color and tonal mass. Semantic vision recovers roughly twice as much
> and improves with model scale (CLIP-B 0.69 → SigLIP-SO400M 0.73) before
> **saturating** — and the remaining gap is largely **label subjectivity**:
> merging near-synonym vibes (introspective/melancholic/haunted) recovers 5–14
> accuracy points, and the most subjective modifier (introspective) resists every
> backbone. Vibe is *distributed rather than cluster-separable* (why an
> unsupervised analysis mistook "no clusters" for "no signal"), and the residual
> is *perceptual overlap*, not missing signal — empirically motivating the
> multimodal `F_obj + F_sem + F_comp` architecture with a human-preference
> `P_vibe` decoder for the irreducibly subjective component.

### Claims to revise
- ❌ Retract: "sentiment is not visually encoded"; "universal k=2 structure";
  "subject isolation is the dominant axis."
- ✅ Keep (reframed): clustering does not recover vibe — because it is entangled
  and non-separable, not absent.
- ✅ Add: supervised floor (F_obj), semantic ceiling (CLIP), complementarity,
  color/tonal localization, subject-independence, scaling-artifact diagnosis.

## Files
- `probe.py` — curated supervised battery (Tests 1–6).
- `bench_fobj.py` — benchmark F_obj probe + image-path resolution.
- `clip_ceiling.py` — CLIP image ceiling (run with `tvm/venv/bin/python`).
- `artifacts/` — cached feature matrices + labels for fast re-runs.

## Reproduce
```bash
python3 strengthening/probe.py
python3 strengthening/bench_fobj.py
/Users/jessicafan/tvm/venv/bin/python strengthening/clip_ceiling.py
```
Note: scripts currently write intermediate arrays to the session scratchpad path;
cached copies are in `artifacts/`.

## TODO (next)
- ✅ done: curated 9-modifier CLIP ceiling (`clip_ceiling_curated.py`); confusion matrix.
- Per-class precision/recall table (which modifiers are reliably detectable vs not).
- Optional: stronger backbone (CLIP ViT-L / DINOv2 / SigLIP) to raise the ceiling.
- Optional: collapse near-synonym labels (introspective/melancholic/haunted) and re-probe.
