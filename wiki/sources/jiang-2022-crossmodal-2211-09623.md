---
type: source
subtype: paper
title: Cross-Modal Adapter for Text-Video Retrieval
slug: jiang-2022-crossmodal-2211-09623
date: 2026-04-20
language: en
tags: [multimodal, retrieval, adapters, video, clip]
processed: true

raw_file: raw/papers/jiang-2022-crossmodal-2211-09623/paper.pdf
raw_md: raw/papers/jiang-2022-crossmodal-2211-09623/paper.md
bibtex_file: raw/papers/jiang-2022-crossmodal-2211-09623/paper.bib
possibly_outdated: true

authors:
  - Haojun Jiang
  - Jianke Zhang
  - Rui Huang
  - Chunjiang Ge
  - Zanlin Ni
  - Jiwen Lu
  - Jie Zhou
  - Shiji Song
  - Gao Huang
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2211.09623
doi:
url: http://arxiv.org/abs/2211.09623
citation_key: jiang2022crossmodal
paper_type: method

read_status: unread
read_date:
rating:

domain: multimodal
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

This paper proposes Cross-Modal Adapter, a parameter-efficient method for adapting CLIP to text-video retrieval without fully fine-tuning both encoders. The model inserts bottleneck adapters into every layer of the video and text transformers and shares part of the adapter up-projection weights across modalities, creating implicit early cross-modal interaction while keeping per-sample encoding independent and `O(N)` rather than `O(N^2)`. A parameter-free similarity calculator then builds query-aware video embeddings by softmax-weighting frame features against the text embedding. Across MSR-VTT, MSVD, VATEX, ActivityNet, and DiDeMo, the method trains fewer than `1M` parameters, reduces fine-tuned parameters by `99.6%` versus CLIP4Clip on MSR-VTT, and reports about `30%` lower training time while maintaining comparable or better retrieval accuracy.

## Problem & Motivation

Fully fine-tuning CLIP-style vision and text encoders is effective for text-video retrieval, but the paper argues that this regime becomes fragile as pretrained backbones scale: it risks overfitting on relatively small downstream datasets and requires training and storing a separate large checkpoint for each benchmark. A naive adapter design could reduce trainable parameters, but independent adapters inside the two encoders prevent any early cross-modal interaction. Directly adding explicit text-video interactions inside shallow layers would require pairwise coupling between every text and every video in a batch, increasing complexity from `O(N)` to `O(N^2)`. The paper therefore targets a parameter-efficient adaptation mechanism that preserves frozen CLIP backbones, enables early multimodal alignment, and avoids quadratic retrieval-time costs.

## Method

- **Backbone encoders**: reuse pretrained CLIP with a `12`-layer ViT-B/32 visual encoder and a `12`-layer text transformer; the text hidden size is `512`, while visual frame features are projected into the text space through `Proj(LN(p^0_{i,j,L}))`.
- **Video representation**: sample frames from each video, encode each frame independently, and keep the frame `[CLS]` embedding as `f_{i,j}` after projection into the shared text-feature dimension.
- **Adapter bottleneck**: start from the standard bottleneck adapter `Adapter(x) = x + \sigma(xW_down)W_up`, where `W_down \in \mathbb{R}^{d \times r}`, `W_up \in \mathbb{R}^{r \times d}`, `r << d`, and `\sigma` is GELU; adapters are initialized near identity.
- **Cross-modal adapter**: split the up-projection into modality-specific and modality-shared blocks so that `Adapter_CM(x) = x + Concat[zW_up,unique, zW_up,share]`, with `z = \sigma(xW_down)` and shared width `d_s`; the adapter is inserted after self-attention and after the MLP in every layer of both encoders.
- **Implicit early interaction**: instead of explicit feature fusion between every text-video pair, the model shares `W_up,share` across modalities, encouraging the two encoders to re-align their internal feature spaces while preserving independent encoding and linear-in-batch computation.
- **Parameter-free similarity calculator**: compute frame-query scores `\alpha_j = <t, f_{i,j}>`, normalize them with temperature `\tau` as `\hat{\alpha}_j = exp(\alpha_j / \tau) / \sum_k exp(\alpha_k / \tau)`, and form a query-aware video embedding `\hat{v}_i = \sum_j \hat{\alpha}_j f_{i,j}`.
- **Training objective**: build the full `n \times n` similarity matrix between `\hat{v}_i` and `t_j`, then optimize symmetric retrieval loss `L = 1/2 (L_v2t + L_t2v)` using cosine similarity.
- **Hyperparameters**: use Adam for `5` epochs on most datasets and `20` epochs on ActivityNet; learning rate `1e-5`, cosine decay, warmup over the first `10%` of steps, batch size `128`, temperature `\tau = 5`, caption length `32` and frame length `12` by default, increased to `64` on ActivityNet and DiDeMo.
- **Dataset-specific adapter sizes**: set bottleneck dimension `r = {8, 8, 16, 16, 16}` and shared width `d_s = {16, 8, 16, 64, 512}` for MSR-VTT, DiDeMo, MSVD, VATEX, and ActivityNet respectively.

## Key Results

- On MSR-VTT, the method uses `0.52M` trainable parameters and reaches text-to-video `R@1/R@5/R@10 = 45.4/73.3/82.3`, outperforming CLIP4Clip with a parameter-free similarity calculator at `44.8/71.2/81.1`.
- Relative to the fully fine-tuned CLIP4Clip setup with `123.52M` updated parameters, the paper reports a `99.6%` reduction in trainable parameters on MSR-VTT with no performance degradation and about `30%` lower training time.
- On MSVD, DiDeMo, VATEX, and ActivityNet, the model stays under `1M` trainable parameters and obtains text-to-video `R@1` of `47.4`, `45.0`, `59.3`, and `41.5`, versus CLIP4Clip at `46.2`, `43.4`, `59.3`, and `41.4`.
- Weight sharing improves over vanilla adapters on all five datasets, with text-to-video `R@1` gains of `+0.5` on MSR-VTT, `+0.7` on MSVD, `+0.4` on VATEX, `+0.3` on ActivityNet, and `+0.9` on DiDeMo.
- The paper reports healthier training dynamics than full fine-tuning on MSR-VTT: CLIP4Clip peaks early and then degrades, whereas the adapter model peaks when training loss is lowest, consistent with reduced overfitting.

## Limitations

- The study only evaluates a single pretrained backbone family, CLIP ViT-B/32, so it does not show whether the same adapter design scales to much larger vision-language encoders.
- Empirical validation is limited to five supervised retrieval benchmarks; there is no zero-shot retrieval, transfer-learning, or non-retrieval multimodal evaluation.
- The gains from the weight-sharing mechanism are real but numerically modest in ablation, mostly between `+0.3` and `+0.9` `R@1`, so the mechanism helps but is not a dramatic architectural jump by itself.
- The method intentionally uses a parameter-free similarity calculator, which keeps tuning cheap, but some richer fully fine-tuned systems still post stronger metrics on particular settings, such as X-CLIP on ActivityNet text-to-video `R@1` (`44.3` vs `41.5`).
- The paper argues overfitting qualitatively and through training curves, but it does not provide a broader analysis across data scales or larger frozen backbones.

## Concepts Extracted

- [[text-video-retrieval]]
- [[parameter-efficient-fine-tuning]]
- [[adapter]]
- [[cross-modal-interaction]]
- [[weight-sharing]]
- [[parameter-free-similarity-calculator]]
- [[query-aware-video-representation]]
- [[prompt-learning]]
- [[overfitting]]
- [[multi-modal-learning]]

## Entities Extracted

- [[haojun-jiang]]
- [[jianke-zhang]]
- [[rui-huang]]
- [[chunjiang-ge]]
- [[zanlin-ni]]
- [[jiwen-lu]]
- [[jie-zhou]]
- [[shiji-song]]
- [[gao-huang]]
- [[tsinghua-university]]
- [[beijing-institute-of-technology]]
- [[msr-vtt]]
- [[msvd]]
- [[vatex]]
- [[activitynet]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
