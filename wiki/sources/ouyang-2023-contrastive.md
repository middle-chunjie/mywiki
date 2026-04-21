---
type: source
subtype: paper
title: Contrastive Learning for Conversion Rate Prediction
slug: ouyang-2023-contrastive
date: 2026-04-20
language: en
tags: [contrastive-learning, cvr-prediction, online-advertising, representation-learning, click-conversion]
processed: true

raw_file: raw/papers/ouyang-2023-contrastive/paper.pdf
raw_md: raw/papers/ouyang-2023-contrastive/paper.md
bibtex_file: raw/papers/ouyang-2023-contrastive/paper.bib
possibly_outdated: true

authors:
  - Wentao Ouyang
  - Rui Dong
  - Xiuwu Zhang
  - Chaofeng Guo
  - Jinmei Luo
  - Xiangzheng Liu
  - Yanlong Du
year: 2023
venue: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval
venue_type: conference
arxiv_id:
doi: 10.1145/3539618.3591968
url: https://dl.acm.org/doi/10.1145/3539618.3591968
citation_key: ouyang2023contrastive
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper proposes CL4CVR, a contrastive-learning framework for post-click conversion rate prediction under extreme label sparsity in online advertising. It pairs a supervised ESMM predictor with a contrastive objective over augmented embedding views so the model can exploit abundant unlabeled click data while still optimizing conversion labels. The central design choice is embedding masking, which preserves all feature fields while randomly masking embedding dimensions, avoiding the information loss caused by feature masking in feature-rich CVR settings. The method also removes duplicate-feature false negatives and adds extra positives from converted examples. On two real-world datasets, CL4CVR consistently outperforms supervised and contrastive baselines, improving CVR AUC by up to `+0.0079` on the industrial benchmark.

## Problem & Motivation

Post-click CVR prediction estimates `p(z = 1 | y = 1, x)` and directly affects ad ranking and charging. The task is difficult because conversion labels are far sparser than clicks: even in very large advertising systems, users click only a small subset of ads and convert on an even smaller subset. Standard deep CVR models therefore become data hungry and under-utilize abundant unlabeled interaction data. The paper asks how to inject a contrastive objective into feature-rich CVR modeling without destroying essential user, item, and context information during augmentation, and how to avoid mislabeled negatives caused by repeated exposures of the same ad-user context.

## Method

- **Supervised backbone**: CL4CVR uses [[esmm]] as the prediction model with shared embeddings, a CTR tower, and a CVR tower. The supervised loss is `L_pred = (1/N) Σ l(ŷ_n, y_n) + (1/N) Σ l(ŷ_n ẑ_n, y_n z_n)`, where the second term optimizes joint click-conversion prediction in the full sample space.
- **Embedding masking (EM)**: instead of masking raw feature fields, the method applies two element-wise random masks to the concatenated embedding vector `e`. If there are `F` feature fields and per-field embedding size `K`, a feature mask has size `F` while an embedding mask has size `F K`, so each augmented view still retains every feature field but with partial embedding dimensions suppressed.
- **Contrastive encoder**: each masked view `ẽ_i` is passed through the same MLP encoder `h_i = f(ẽ_i)`. The encoder uses fully connected layers with sizes `{512, 256, 128}` and ReLU on all but the last layer.
- **Base contrastive objective**: with `2N` augmented samples from a mini-batch of size `N`, the SimCLR-style loss is `L_0 = -(1/(2N)) Σ_i log (exp(s(h_i, h_j)/τ) / Σ_{k ≠ i} exp(s(h_i, h_k)/τ))`, where `s(·,·)` is cosine similarity and `τ` is a temperature hyperparameter.
- **False negative elimination (FNE)**: repeated impressions can yield multiple samples with identical features but different conversion outcomes, so treating them as negatives is contradictory. The paper defines `M(i) = {j} ∪ {k | I(o(ẽ_i), o(ẽ_k)) = 0}` to exclude duplicates from the denominator while keeping all samples for supervised learning.
- **Supervised positive inclusion (SPI)**: for anchor `i`, the positive set is expanded to `S(i) = {j} ∪ {k | z(ẽ_k) = z(ẽ_i) = 1, k ≠ i, k ≠ j}`. Only label-1 examples contribute extra positives, avoiding the degenerate case where numerous label-0 samples eliminate contrast.
- **Overall training setup**: the final loss is `L = L_pred + α L_cl`, where `L_cl` averages over `Q(i) = S(i) ∩ M(i)`. Experiments use `batch_size = 64`, TensorFlow implementation, Adagrad optimization, and 3 runs per method. Hyperparameter sweeps show that larger `τ` works better on both datasets, while overly large `α` hurts by overweighting the contrastive task.

## Key Results

- On the industrial dataset, CL4CVR reaches CVR AUC `0.8637`, beating the supervised ESMM base (`0.8558`) by `+0.0079` and the best non-CL baseline SO (`0.8563`) by `+0.0074`.
- On the public Taobao-derived dataset, CL4CVR reaches CVR AUC `0.6590`, improving over the base (`0.6524`) by `+0.0066`.
- Simple feature dropout degrades performance substantially: industrial AUC drops from `0.8558` to `0.8452`, and public AUC drops from `0.6524` to `0.6469`.
- Embedding masking alone is already stronger than prior masking baselines: industrial/public AUCs `0.8586 / 0.6572` versus RFM `0.8522 / 0.6536` and CFM `0.8539 / 0.6541`.
- Adding FNE to EM yields `0.8605 / 0.6581`; adding SPI to EM yields `0.8617 / 0.6580`; combining all three components yields the best results, indicating complementary gains from duplicate handling and supervised positives.
- Dataset scale highlights the sparsity regime: the industrial split has `278.8M / 49.2M / 48.4M` train/val/test samples with only `0.67M` conversions, while the public dataset has `2.3M / 0.98M / 3.3M` with `0.018M` conversions.

## Limitations

- The paper evaluates only two advertising datasets from closely related production settings, so transfer beyond CVR prediction or beyond Alibaba-style ad logs is untested.
- Core hyperparameters such as the embedding-mask rate, temperature `τ`, and loss weight `α` are discussed qualitatively, but the paper does not provide the full search space or exact best settings in the main text.
- The method depends on detecting duplicated feature vectors for FNE; the operational cost and robustness of this duplicate check in larger or noisier systems are not analyzed.
- The supervised model is fixed to ESMM, so it is unclear how much of the gain transfers to stronger modern CVR architectures or to delayed-feedback and multi-task formulations.
- As an IR-adjacent 2023 paper in a fast-moving area, some design choices should be rechecked against newer recommendation and representation-learning literature.

## Concepts Extracted

- [[contrastive-learning]]
- [[conversion-rate-prediction]]
- [[embedding-masking]]
- [[false-negative-elimination]]
- [[supervised-positive-inclusion]]
- [[feature-masking]]
- [[contrastive-loss]]
- [[representation-learning]]
- [[data-augmentation]]

## Entities Extracted

- [[wentao-ouyang]]
- [[rui-dong]]
- [[xiuwu-zhang]]
- [[chaofeng-guo]]
- [[jinmei-luo]]
- [[xiangzheng-liu]]
- [[yanlong-du]]
- [[alibaba-group]]
- [[esmm]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
