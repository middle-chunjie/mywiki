---
type: source
subtype: paper
title: "CrossCLR: Cross-modal Contrastive Learning For Multi-modal Video Representations"
slug: zolfaghari-2021-crossclr-2109-14910
date: 2026-04-20
language: en
tags: [contrastive-learning, cross-modal, video-text-retrieval, multimodal, representation-learning]
processed: true
raw_file: raw/papers/zolfaghari-2021-crossclr-2109-14910/paper.pdf
raw_md: raw/papers/zolfaghari-2021-crossclr-2109-14910/paper.md
bibtex_file: raw/papers/zolfaghari-2021-crossclr-2109-14910/paper.bib
possibly_outdated: true
authors:
  - Mohammadreza Zolfaghari
  - Yi Zhu
  - Peter Gehler
  - Thomas Brox
year: 2021
venue: arXiv
venue_type: preprint
arxiv_id: "2109.14910"
doi:
url: http://arxiv.org/abs/2109.14910
citation_key: zolfaghari2021crossclr
paper_type: method
read_status: unread
domain: multimodal
---

## Summary

> ⚠️ **Possibly outdated** — published 2021 in the volatile multimodal/video-language domain; verify against post-2023 literature before relying on benchmark comparisons.

CrossCLR proposes a contrastive loss for learning joint multi-modal embeddings that addresses two systematic flaws in prior cross-modal contrastive learning: (1) the absence of intra-modality alignment constraints, which allows semantically similar items within the same modality to drift apart in the joint embedding; and (2) the uncritical inclusion of semantically similar items as negatives (false negatives), which causes "semantic collision." The solution adds intra-modality loss terms and introduces "influential samples" — high-connectivity items pruned from the negative set and up-weighted in the loss. Built on the COOT hierarchical architecture with BERT and pre-trained visual experts, CrossCLR advances state-of-the-art on video-text retrieval on Youcook2 and LSMDC and on video captioning on Youcook2.

## Problem & Motivation

Standard cross-modal contrastive losses (MaxMargin, InfoNCE/NT-Xent, CLIP-style) only enforce inter-modality alignment: pulling a video embedding close to its matching text. Two gaps remain:

1. **Missing intra-modality alignment**: nothing prevents two semantically similar videos (or two similar texts) from diverging in the joint space. Prior works implicitly assumed transitivity would handle this; CrossCLR shows it does not.
2. **False negative problem (semantic collision)**: in large minibatches, many negative pairs share genuine semantic overlap (e.g., "cut tomatoes and mix with herbs" vs. "cut tomato and put in a bowl"). Treating them as hard negatives actively degrades representation quality by discarding their shared features.

These issues are especially acute when input modalities already carry pre-trained semantic embeddings, as in the "expert features + small joint-embedding network" paradigm.

## Method

### Architecture

CrossCLR reuses the COOT two-level hierarchical transformer:
- **Local transformer**: maps per-frame expert features and per-word BERT features to clip/sentence embeddings.
- **Global transformer**: aggregates clip/sentence embeddings into video/paragraph embeddings.
- Loss applied at both levels: `L = L_local + 0.6 * L_global`.
- Visual experts: ResNeSt269 (appearance/ImageNet), DenseNet161 (scene/Places365), R(2+1)D-ResNet152 (action/IG65M), Faster-RCNN-ResNet50-FPN (objects), HowTo100M features (for Youcook2).
- Text encoder: BERT-Base uncased (frozen).

### CrossCLR Loss

**Notation**: `δ(x_i, y_j) = exp(z_{x_i}^T z_{y_j} / τ)` with `l2`-normalised projections. Temperature `τ = 0.03`.

**Inter + intra alignment (without pruning, Eq. 2–3)**:

```
L(x_i) = -log [ δ(x_i, y_i) / ( δ(x_i, y_i)
               + Σ_{y_k ∈ N^E_i} δ(x_i, y_k)      # inter-modal negatives
               + λ Σ_{x_j ∈ N^R_i} δ(x_i, x_j) ) ] # intra-modal negatives
```

`λ` controls the intra-modality alignment strength (`λ = 8e1` for Youcook2, `λ = 65e2` for LSMDC).

**Influential samples** (Eq. 4): connectivity of `x_i` over a momentum queue `Q` of size `M`:

```
C(x_i) = (1/M) Σ_j  (x_i^T x_j) / (||x_i|| ||x_j||)
```

Threshold `γ = 0.9`; samples with `C(x_i) > γ` are "influential" and placed in `I_x`.

**Negative set pruning**: influential samples are removed from both inter-modal (`N^E`) and intra-modal (`N^R`) negative sets.

**Proximity weighting** (Eq. 5): per-sample loss weight `w(x_i) = exp(C(x_i) / κ)` with `κ = 35e4` (Youcook2) / `κ = 55e4` (LSMDC). Up-weights influential samples, down-weights outliers.

**Final loss**: `L = (L_x + L_y) / 2` where `L_x` and `L_y` are the weighted pruned losses over each modality (Eq. 6–7).

**Queue**: `3K` (Youcook2) / `5K` (LSMDC); updated by FIFO replacement each minibatch.

**Optimizer**: RAdam, lr = `7e-4`, momentum = `0.56`, warmup 4 epochs, ReduceOnPlateau patience = 6. Trained 40 epochs (~1h Youcook2, ~2h LSMDC on 1× NVIDIA T4).

## Key Results

**Table 1 — Contrastive loss comparison (same COOT architecture), Youcook2 Text→Video**:
- MaxMargin: R@1 = 15.0, R@5 = 37.0, R@10 = 49.1
- MIL-NCE: R@1 = 18.0, R@5 = 41.9, R@10 = 53.9
- CLIP: R@1 = 17.8, R@5 = 42.1, R@10 = 54.4
- **CrossCLR: R@1 = 19.5, R@5 = 45.9, R@10 = 58.3** (+1.5 pp R@1 over best baseline)

**Table 3 — SOTA on Youcook2 Text→Video**:
- COOT: R@1 = 16.7; **CrossCLR: R@1 = 19.5** (new SOTA at publication)

**Table 4 — SOTA on LSMDC Text→Video**:
- MMT: R@1 = 13.2, MdR = 21.0; **CrossCLR: R@1 = 15.0, MdR = 18.0** (fewer modalities than MMT which uses Audio, OCR, Face)

**Table 6 — Video captioning (Youcook2 val)**:
- COOT: CIDEr-D = 54.07; **CrossCLR: CIDEr-D = 58.65**; with video-level embeddings: **61.10** vs COOT+ 57.24

**Ablation (Table 2)**: all three components contribute; intra-modality alignment (I_M) gives the largest single gain; negative pruning (N_P) adds further ~0.5 pp R@1.

**Cross-modality generalization (Table 5)**: CrossCLR beats NT-Xent across nearly all 5×5 modality pairs on LSMDC, confirming the loss is not specific to video-text.

## Limitations

- Relies on pre-trained expert features that are kept frozen; performance depends heavily on the quality of the visual expert set.
- The influential-sample threshold `γ` and the queue size require dataset-specific tuning; no adaptive strategy is proposed.
- Connectivity-based influential sample identification is a heuristic proxy for false-negative detection — it does not directly measure semantic overlap.
- Experiments limited to two relatively small datasets (Youcook2 ~9.6K clips, LSMDC ~118K clips); scalability to web-scale (e.g., HowTo100M pre-training) is not evaluated.
- The paper predates the explosion of large vision-language pre-training (e.g., CLIP at scale, ALIGN, Florence, InternVideo); results may not transfer to that regime.

## Concepts Extracted

- [[contrastive-learning]]
- [[cross-modal-alignment]]
- [[multimodal-contrastive-learning]]
- [[contrastive-loss]]
- [[false-negative-elimination]]
- [[false-negative]]
- [[in-batch-negative-sampling]]
- [[negative-sampling]]
- [[hierarchical-transformer]]
- [[text-video-retrieval]]
- [[joint-embedding]]
- [[intra-modal-alignment]]
- [[influential-sample]]
- [[video-captioning]]

## Entities Extracted

- [[mohammadreza-zolfaghari]]
- [[yi-zhu-amazon]]
- [[peter-gehler]]
- [[thomas-brox]]
- [[university-of-freiburg]]
- [[amazon-tubingen]]
- [[youcook2]]
- [[lsmdc]]
- [[howto100m]]
- [[coot]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
