# TVM — revised Results & Discussion (drop-in draft)

Replaces the clustering-only treatment with a supervised, multi-method analysis.
All numbers from `strengthening/` (probe.py, bench_fobj.py, clip_ceiling*.py,
ceiling_backbones.py, ceiling_openclip.py, collapse_test.py, collapse_bench.py).

---

## Methods addendum (new experiments)

We complement the unsupervised analysis with supervised *probes*, which test
directly whether a representation contains label-relevant information rather than
whether labels happen to form clusters. For each feature space we train a
classifier (multinomial logistic regression and gradient-boosted trees) under
5-fold stratified cross-validation and report **balanced accuracy** and
**macro-F1** (the datasets are class-imbalanced; balanced accuracy corrects for
this). A **label-permutation test** (re-running the probe on shuffled labels)
establishes the empirical chance floor. To separate vibe from scene content we
additionally evaluate under **GroupKFold by subject**, so no subject appears in
both train and test. For the semantic representation we extract image embeddings
from frozen vision–language encoders (OpenAI CLIP ViT-B/32, ViT-B/16, ViT-L/14;
SigLIP ViT-SO400M-14-384) and apply the same probe. Finally, a
**taxonomy-collapse** analysis greedily merges the most-confused label pair and
re-probes, tracing how much residual error is attributable to category overlap.

---

## Results (revised)

### R1. Aesthetic vibe is recoverable from objective composition
Unsupervised clustering does not recover the modifier labels (Sec. 5), but this
reflects the limits of clustering, not the absence of signal. A supervised probe
on the 41-dimensional objective feature space `F_obj` recovers the 9 modifiers
well above chance:

| Dataset | balanced acc | macro-F1 | chance (permuted) | ratio |
|---|---|---|---|---|
| Curated (4,994 img, 9 modifiers) | 0.451 | 0.463 | 0.113 | 4.0× |
| Benchmark (812 img, 8 emotions) | 0.335 | 0.333 | 0.126 | 2.7× |

The effect is robust to the subject confound: predicting modifiers under
GroupKFold by subject (subjects disjoint between train/test) yields balanced
accuracy 0.421 on the curated set — essentially unchanged — so the signal is a
property of vibe, not of the underlying scene. The probe predicts *subject*
(26-way) at 0.278, confirming `F_obj` also carries scene information, but the two
are separable.

### R2. The signal is color and tonal, not framing
Per-feature ANOVA across modifiers and mutual-information ranking agree that the
most discriminative dimensions are color/tonal/visual-mass:
`mean_visual_weight` (F=417), `warm_color_ratio` (322), `shadow_ratio` (184),
`midtone_ratio`, `mean_contrast`, `cool_color_ratio`. The subject-isolation and
framing features that dominate the binary clustering split do *not* drive
modifier separation; they drive an unrelated outlier axis (R3).

### R3. The "k=2 / silhouette 0.99" structure is a scaling artifact
Three features are unbounded after robust scaling (max |z| up to 6.2×10⁴), so a
handful of extreme values dominate Euclidean distance and k-means trivially
isolates ~30 outliers, yielding silhouette ≈ 0.994. Under correct preprocessing
(1–99% winsorization + standardization) the silhouette of the k=2 solution
collapses to 0.119, and no strong cluster structure remains at any k. The
cross-dataset "universality" of k=2 reflects this shared preprocessing pathology
rather than a shared compositional axis. Cluster–label agreement is near zero
throughout (NMI 0.00–0.07), confirming that modifiers are **distributed and
non-separable** in `F_obj` — present to a classifier, invisible to clustering.

### R4. Semantic vision recovers ~2× more, and scale saturates
Probing frozen image embeddings on the same images (curated, chance 0.111):

| Representation | balanced acc | macro-F1 |
|---|---|---|
| F_obj (floor) | 0.448 | — |
| CLIP ViT-B/32 | 0.687 | 0.686 |
| CLIP ViT-B/16 | 0.687 | 0.689 |
| CLIP ViT-L/14 | 0.707 | 0.708 |
| SigLIP-SO400M-384 | 0.731 | 0.729 |
| F_obj + SigLIP | 0.734 | — |

Semantic vision roughly doubles the recoverable signal over hand-crafted
composition. Performance rises with model scale but saturates (+0.044 from
ViT-B to SigLIP, diminishing per step). Notably, `F_obj` is complementary only
to *weak* backbones: it adds +0.012 on top of CLIP-B/32 but nothing measurable on
top of SigLIP — i.e., a strong semantic encoder subsumes the interpretable
composition signal, while remaining far from perfect.

### R5. The recoverable structure is uneven across vibes
Per-modifier F1 (curated), F_obj floor vs SigLIP ceiling:

| Modifier | F_obj F1 | SigLIP F1 | reading |
|---|---|---|---|
| radiant | 0.623 | 0.774 | color-driven; composition nearly competitive |
| melancholic | 0.526 | 0.731 | tonal; composition decent |
| nostalgic | 0.372 | 0.749 | needs semantics |
| romantic | 0.447 | 0.817 | needs semantics |
| candid | 0.295 | 0.766 | relational/semantic |
| whimsical | 0.425 | 0.873 | needs semantics |
| ethereal | 0.430 | 0.667 | partly subjective |
| haunted | 0.411 | 0.733 | needs semantics |
| introspective | 0.213 | 0.451 | subjective; hard for every backbone |

Composition is most informative for vibes defined by color and tone (radiant,
melancholic) and least informative for relational or narrative vibes (candid,
whimsical, romantic), which require semantic grounding.

### R6. The residual gap is label subjectivity, not model capacity
Greedily merging the most-confused (near-synonymous) labels and re-probing with
SigLIP recovers accuracy rapidly:

Curated: 9→8 (introspective+melancholic) 0.733→0.784; →7 (+haunted) 0.813; →6
(ethereal+radiant) 0.819; →5 (+romantic) 0.858; →4 (+nostalgic) 0.875.
Benchmark: 8→7 (fear+sad) 0.622→0.676; →5 (amusement+excitement) 0.756; →3
(awe+contentment) 0.829.

A single near-synonym merge recovers ~5 points on each dataset, and the merge
order follows recognizable affective structure (negative-valence fear/sad;
positive/high-arousal amusement/excitement) consistent with a valence–arousal
manifold. Much of the remaining error is therefore **perceptual overlap between
adjacent categories**, not a failure to encode vibe. The most subjective
modifier, *introspective*, is not perceptually distinct from *melancholic* and
resists every representation.

---

## Discussion (revised points)

- **Reinterpretation.** Aesthetic vibe *is* visually encoded — partially in
  objective composition (color/tone) and substantially in semantic vision — but
  it is distributed rather than cluster-separable. The earlier conclusion that
  sentiment "cannot be recovered from objective features" conflated *not
  clusterable* with *not present*; a supervised probe recovers it at up to 4×
  chance, and it survives subject control.

- **Architecture.** The results give an empirical, quantitative basis for the
  multimodal design. `F_obj` is a cheap, interpretable, deterministic backbone
  that captures the color/tonal component (and is complementary to weak
  encoders); `F_sem` (a strong VLM) captures the relational/semantic component
  and roughly doubles recoverable signal; `F_comp` learns their combination; and
  `P_vibe` — trained on human preference — is required for the irreducibly
  subjective residual, which scaling alone does not close.

- **Where the ceiling comes from.** Scaling the semantic encoder saturates near
  0.73 (9-way). The collapse analysis shows much of the gap to perfect is
  taxonomy overlap; the rest is genuine subjectivity (introspective). This
  bounds what *any* purely visual model can achieve on this labeling and
  motivates either a coarser, perceptually grounded taxonomy or multi-annotator
  soft labels.

- **Limitations (updated).** Labels are single-annotator and overlap
  perceptually (quantified in R6); a valence–arousal or multi-label scheme would
  likely raise the achievable ceiling. The probe ceiling is a *lower bound* on
  recoverable signal (frozen encoder + linear/GBM readout); fine-tuning could
  recover more. `F_obj`'s incremental value is demonstrated only against frozen
  encoders.

---

## Claims to revise in the paper
- ❌ Remove: "aesthetic sentiment is not visually encoded / cannot be recovered
  from objective features"; "universal k=2 compositional-formality structure";
  "subject isolation is the dominant discriminative axis."
- ✅ Keep (reframed): unsupervised clustering does not recover vibe — because it
  is distributed and overlapping, not absent.
- ✅ Add: supervised recoverability (R1), subject-independence (R1), color/tonal
  localization (R2), scaling-artifact diagnosis of k=2 (R3), semantic ceiling and
  saturation (R4), per-vibe heterogeneity (R5), subjectivity-vs-capacity
  decomposition (R6).
