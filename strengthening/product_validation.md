# Product-mechanic validation (startup / license thesis)

Tests the two open bets behind a "vibe layer" business (`retrieval_taste.py`).
Strategy: B2B visual-commerce as the vehicle; data + taxonomy as the asset;
license/partner an incumbent as the likely route. Model is NOT the moat (proven:
SigLIP = B/32 at 30 vibes) — so an acquirer buys validated capability + preference data.

## A — "Shop the vibe" retrieval precision@k (does NN retrieval return same-vibe items?)

| set / backbone | P@1 | P@5 | P@10 | random |
|---|---|---|---|---|
| curated / SigLIP | **0.706** | 0.648 | 0.606 | 0.137 |
| curated / B32 | 0.658 | 0.603 | 0.566 | 0.137 |
| scraped 30-vibe / SigLIP | 0.304 | 0.214 | 0.175 | 0.033 |
| scraped **super-vibe (5)** / SigLIP | **0.566** | 0.491 | 0.461 | 0.265 |

- On validated data the core mechanic works: given an inspiration image, ~71% of the
  top hit and ~61% of the top-10 share its vibe (~5× random).
- Fine 30-vibe retrieval is weak (overlap), but at the **granularity a product would
  actually use (super-vibes)** it is strong (0.57 P@1). Ship at super-vibe granularity.
- **Concrete vibes are retrieval-ready; abstract are not** (scraped/SigLIP P@10):
  brutalist .57, futuristic .44, abandoned .31 vs dreamy .09, silent .10, twinkling .11.
- **Per curated modifier (SigLIP P@10):** romantic .72, whimsical .71, melancholic .70,
  radiant .64, haunted .60, candid .57, ethereal .56, nostalgic .46, **introspective .32**.
  → 6–7 of 9 are production-ready; introspective/nostalgic deferred (V1 ship-list).

## B — Few-shot taste learning (learn a vibe from k "likes", rank the rest; mean AP)

| set | k=1 | k=3 | k=5 | k=10 | random |
|---|---|---|---|---|---|
| curated / SigLIP | 0.251 | 0.301 | 0.307 | 0.329 | 0.137 |
| scraped / SigLIP | 0.052 | 0.054 | 0.055 | 0.059 | 0.033 |

- Taste learns **fast**: a single liked image ~doubles AP over random; gains plateau by
  k≈3–5. The flywheel needs little per-user signal to start personalizing.
- Fine 30-vibe few-shot is weak (single-label proxy + heavy overlap); real "taste" is a
  region/mixture, so these numbers under-state a well-designed system.

## Read
- ✅ Moat (curation ≫ scraping), ✅ retrieval mechanic works (clean data + super-vibe
  granularity), ✅ taste learns from 1–3 examples, ✅ concrete-first roadmap.
- ❓ **Unprovable with offline data:** (a) vibe match → actual purchase/engagement lift;
  (b) inspiration → *product catalog* match (we tested image→image, not query→product).
  Both require a real catalog + behavioral data — i.e., a **B2B design partner**, not more
  offline analysis. This is the natural stopping point for desk research.

## License-ready one-liner
"Vibe retrieval works (P@1≈0.71 on validated data, ~0.57 at product granularity), taste is
learnable from ~3 examples, and the labels can't be scraped (0.93 curated vs 0.54 scraped) —
they require our curation pipeline. The remaining unknown (behavioral lift) is what a design-
partner pilot proves."

## V1 generative test — color/tone re-blend (`reblend_poc.py`, demo `reblend_demo.png`)
Can product-safe color/tone grading move a "product" photo's vibe toward a target feed?
Source=radiant (bright/product-like), target=melancholic (moody). CLIP-B/32, 36 grades/img.

| metric | value |
|---|---|
| native melancholic band (cos-dist) | 0.188 |
| radiant baseline gap | 0.246 |
| after color grading | 0.193 (**86% of gap closed**) |
| reached native band | 47% of images |
| P(melancholic) before→after | 0.04 → 0.15 |
| classifier flips to melancholic | 0% → 16% |

→ Color/tone grading closes most of the *geometric* gap and visibly shifts mood while
preserving content, but only fully converts vibe 16% of the time — because semantic content
carries ~2× more vibe than color (consistent with R4/R5). **Two tiers:** V1 ambiance
*harmonizer* (ships now, product-safe, soft blend) vs V2 generative scene adaptation (full
blend, needs identity-preserving diffusion + fidelity guardrails). The interpretable feature
space is a measurable control + audit surface for both. Caveats: radiant↔melancholic gap is
small in CLIP space; PoC uses global grading (no foreground segmentation yet).

## V1 sharpened + V2 generative (`reblend_v1.py`, `reblend_v2.py`)
**V1 (color/tone, product-safe).** Multi-target (radiant→): ethereal gap-closed 51%/flip 23%,
melancholic 80%/15%, haunted 82%/5% — color closes the geometric gap but conversion drops with
semantic contrast. Fidelity guardrail: structure always preserved (grad-SSIM 0.93–1.0); only
COLOR moves (ΔE 0→32) → true-color is the bounded lever, shape is never at risk. Foreground-lock
demo: background re-vibed, product region pixel-true (ΔE=0).

**V2 (SD img2img) vs V1, same images:**
| target | method | flip | structure | colorΔE |
|---|---|---|---|---|
| melancholic (tonal) | V1 | 10% | 0.919 | 25.2 |
| | V2 | 10% | 0.565 | 10.8 |
| haunted (semantic) | V1 | 5% | 0.905 | 26.5 |
| | V2 | **25%** | 0.557 | 10.9 |

→ Tonal vibes: V2 is pure loss (same flip, structure collapses) — use V1. Semantic vibes: V2
breaks the wall (5×) but only by changing content (structure 0.56) — for product photos that
means the **real V2 = identity-locked generation** (segment product, regenerate background only).
Full vibe conversion is partial (~25% max): blending in is real but soft, not invisibility.
Demos: reblend_v1_fglock.png, reblend_v2_demo.png.
