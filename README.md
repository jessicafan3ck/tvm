# Tiered Vibe Mapping (TVM): A Feature-Space Decomposition for Aesthetic Modeling

A comprehensive Python framework for clustering, dimensionality reduction, and analysis of visual aesthetic sentiment across a dynamic latent space. Decomposes aesthetic nuance using interpretable objective features and validates reliable reconstruction through multimodal embeddings.

##  Overview

This project implements an end-to-end pipeline for understanding visual aesthetics through the lens of latent space decomposition. Given a dataset of 4,994 images annotated across 9 aesthetic modifiers and 24 subject categories, the pipeline:

1. **Extracts objective visual features** (F_obj) from geometric, spatial, color, and compositional dimensions
2. **Applies dimensionality reduction** using PCA, UMAP, and t-SNE
3. **Performs multi-method clustering** (K-means, Hierarchical, DBSCAN) with automatic optimization
4. **Validates latent space structure** (Z_vibe) through separation analysis and reconstruction metrics
5. **Tests compositional properties** to understand modifier × subject interactions

## Central Research Question

**Can visual aesthetic sentiment and nuance be reliably decomposed and reconstructed across a dynamic aesthetic latent space using an interaction between interpretable objective features and learned multimodal embeddings?**

## Dataset

- **Total Images**: 4,994
- **Aesthetic Modifiers**: 9 unique categories
- **Subject Categories**: 24 unique subjects
- **Per Pairing**: ~30 images per modifier-subject combination (ranged from 1 - 50)
- **Feature Extraction**: 7 categories of visual features (~120 total dimensions)

##  Model Architecture

```
Input Image
    ↓
F_obj (Objective Feature Extractor)
    ├─ Geometric & Structural Features
    ├─ Spatial Layout Features
    ├─ Contrast & Color Features
    ├─ Motion & Dynamics Features
    ├─ Pattern & Rhythm Features
    ├─ Visual Mass & Focus Features
    └─ Framing & Breathing Features
    ↓
X_obj (~120 dimensions)
    ↓
[Dimensionality Reduction: PCA → UMAP]
    ↓
Z_vibe (Aesthetic Latent Space, 20-D)
    ↓
P_vibe (Aesthetic Sentiment Decoder)
    ↓
Modifier Prediction (26 categories)
```

**Proposed Multimodal Extension**:
```
Input Image
    ↓
F_sem (Semantic Encoder, e.g., CLIP)
    ↓
X_sem (512-D embeddings)
    ↓
[Fusion: concatenate(X_obj, X_sem)]
    ↓
[Dimensionality Reduction]
    ↓
Z_vibe (Multimodal Latent Space)
```

## Contributors

- Jessica Fan (Primary Author)
- Inae Kim (Data Collector)
- Ivan Zheng (Data Collector)
- Victoria Fan (Data Collector)
- Henry Chen (Data Collector)

## Links

- **GitHub**: https://github.com/jessicafan3ck/tvm/new/main?filename=README.md
- **Overleaf**: https://www.overleaf.com/project/68f51faa8e29bd27694950f5
