---
type: source
subtype: paper
title: What Makes Good Examples for Visual In-Context Learning?
slug: zhang-nd-what
date: 2026-04-20
language: en
tags: [visual-in-context-learning, prompt-retrieval, computer-vision, contrastive-learning, multimodal]
processed: true
raw_file: raw/papers/zhang-nd-what/paper.pdf
raw_md: raw/papers/zhang-nd-what/paper.md
bibtex_file: raw/papers/zhang-nd-what/paper.bib
possibly_outdated: true

authors:
  - Yuanhan Zhang
  - Kaiyang Zhou
  - Ziwei Liu
year: 2023
venue: ICML 2023
venue_type: conference
arxiv_id:
doi:
url:
citation_key: zhangndwhat
paper_type: method

read_status: unread
read_date:
rating:

domain: multimodal
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature in visual in-context learning.

This paper presents the first comprehensive study of how the choice of in-context examples affects performance in visual in-context learning — an emergent capability of large vision models that solves unseen tasks by conditioning on prompt image–label pairs without parameter updates. The authors find that performance is highly sensitive to example selection, with gaps exceeding 70% mIoU between best and worst prompts. They propose a prompt retrieval framework with two implementations: an unsupervised method using off-the-shelf CLIP features for nearest-neighbor search, and a supervised method that fine-tunes a feature extractor via contrastive learning to directly optimize in-context learning performance. Both outperform random selection, with the supervised method achieving the best results across foreground segmentation, single object detection, and image colorization tasks.

## Problem & Motivation

Large vision models accessed via Model-as-a-Service (MaaS) APIs cannot be fine-tuned directly; in-context learning is therefore the primary adaptation mechanism. However, the existing approach of random within-class example selection leads to enormous performance variance (>70% mIoU gap on Pascal-5^i). There is no principled method for selecting good visual in-context examples, and the properties that make an example useful for visual tasks are not yet understood.

## Method

### Framework

Given a dataset `D = {(x_n, y_n)}`, a query `x_q`, and a frozen large vision model `g_τ`, in-context learning is: `y_q = g_τ(P, x_q)` where `P = {x_c1, y_c1, ..., x_cK, y_cK}` is a prompt of `K` image–label pairs. The goal is to select `x* = argmax_{x_n ∈ D} f_θ(x_n, x_q)` where `f_θ` scores the suitability of an example.

`f_θ` is implemented as a neural network encoder followed by cosine distance between feature vectors.

### Unsupervised Prompt Retrieval (UnsupPR)

- Uses a frozen off-the-shelf feature extractor (default: CLIP's vision encoder).
- Scores each training example by cosine similarity to the query in feature space.
- Top-scoring examples form the prompt; no task-specific training required.

### Supervised Prompt Retrieval (SupPR)

- Makes the feature extractor learnable via contrastive training.
- For each training example `x_n`, computes predictions using every other `x_m ∈ D` as a prompt; top-5 and bottom-5 by task metric form positive/negative sets.
- Contrastive loss: `ℓ = -(1/|B|) Σ_{x_n ~ B} log [exp(cos(z_n, z_n+)) / (exp(cos(z_n, z_n+)) + Σ_{z_n- ∈ N} exp(cos(z_n, z_n-)))]`
- Negative set `N` contains one sampled hard negative plus in-batch negatives.
- Training: 200 epochs, SGD, initial lr `0.005`, cosine annealing decay.
- Base model: image inpainting model from Bar et al. (2022), pre-trained on academic figures.

### Evaluation Tasks (unseen during pre-training)

1. **Foreground segmentation**: Pascal-5^i (4 splits, 5 categories each); metric: mIoU.
2. **Single object detection**: Pascal VOC; metric: mIoU.
3. **Image colorization**: ImageNet-2012 (50k validation images); metric: MSE.

## Key Results

- **Main results** (Table 1):
  - Segmentation (avg mIoU): Random 27.56 → UnsupPR 33.56 → SupPR **35.56** (+8.0 over Random)
  - Detection (mIoU): Random 25.45 → UnsupPR 26.84 → SupPR **28.22**
  - Colorization (MSE ↓): Random 0.67 → UnsupPR 0.63 → SupPR **0.63**
- **Distribution shift** (Pascal → MSCOCO, Table 2): SupPR 19.95 vs Random 16.78 avg mIoU; SupPR still outperforms despite being learning-based.
- **Backbone ablation** (Table 3): CLIP, EVA, and ViT backbones differ by <1% for both UnsupPR and SupPR — backbone choice matters little.
- **Order of examples** (Table 4): standard deviation is small; ordering is not a major concern if good examples are chosen.
- **Retrieval set size**: both retrieval methods benefit from more candidates, but SupPR plateaus early — 20% of full training data is sufficient.
- **Number of examples**: more in-context examples consistently improve all three methods.
- **Distance metric**: cosine, Euclidean, and Manhattan distances yield similar results.
- Qualitative analysis shows SupPR retrieves examples that are closer to the query in semantics, background, object pose, appearance, and viewpoint than UnsupPR.

## Limitations

- Evaluated only on the Bar et al. (2022) image inpainting backbone; results may not generalize to other visual in-context learning architectures.
- Distribution shift degrades performance substantially: SupPR gains shrink from +8% to +3% mIoU when training on Pascal and testing on MSCOCO, suggesting the feature extractor does not generalize perfectly.
- Colorization gains are marginal, possibly because the base image inpainting model is inherently weak at this task — the retrieval method cannot compensate for model-level limitations.
- The paper does not provide a theoretical explanation for why semantically and spatially similar examples are good prompts.
- The bib entry lacks year metadata (slug uses "nd"); the year is inferred as ~2023 from the ICML venue and citation dates.
- No exploration of multi-modal or cross-task prompt construction strategies.

## Concepts Extracted

- [[visual-in-context-learning]]
- [[prompt-retrieval]]
- [[in-context-learning]]
- [[contrastive-learning]]
- [[contrastive-loss]]
- [[image-inpainting]]
- [[model-as-a-service]]
- [[foundation-model]]
- [[cosine-similarity]]
- [[nearest-neighbor-retrieval]]
- [[few-shot-learning]]
- [[visual-prompting]]

## Entities Extracted

- [[yuanhan-zhang]]
- [[kaiyang-zhou]]
- [[ziwei-liu]]
- [[nanyang-technological-university]]
- [[clip]]
- [[gpt-3]]
- [[imagenet]]
- [[ms-coco]]
- [[pascal-voc]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
