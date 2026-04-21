---
type: source
subtype: paper
title: Meta-optimized Contrastive Learning for Sequential Recommendation
slug: qin-2023-metaoptimized
date: 2026-04-20
language: en
tags: [sequential-recommendation, contrastive-learning, meta-learning, recommender-systems, representation-learning]
processed: true

raw_file: raw/papers/qin-2023-metaoptimized/paper.pdf
raw_md: raw/papers/qin-2023-metaoptimized/paper.md
bibtex_file: raw/papers/qin-2023-metaoptimized/paper.bib
possibly_outdated: true

authors:
  - Xiuyuan Qin
  - Huanhuan Yuan
  - Pengpeng Zhao
  - Junhua Fang
  - Fuzhen Zhuang
  - Guanfeng Liu
  - Yanchi Liu
  - Victor Sheng
year: 2023
venue: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval
venue_type: conference
arxiv_id:
doi: 10.1145/3539618.3591727
url: https://dl.acm.org/doi/10.1145/3539618.3591727
citation_key: qin2023metaoptimized
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

MCLRec is a sequential recommendation method that augments standard contrastive learning with learnable model augmenters and a meta-optimization loop. Starting from stochastic data augmentations such as mask, crop, and reorder, the method feeds two augmented sequence representations into two MLP augmenters to create additional contrastive views without increasing batch size. Training alternates between updating the encoder with recommendation loss plus two contrastive losses, and updating the augmenters with a meta stage driven by encoder performance and a contrastive regularizer that discourages collapsed or overly similar views. On Amazon Sports, Amazon Beauty, and Yelp, MCLRec consistently outperforms strong sequential recommendation baselines including DuoRec and SRMA, with relative gains of 3.94% to 8.41% on HR and 2.47% to 5.69% on NDCG.

## Problem & Motivation

Sequential recommendation models such as GRU4Rec, [[sasrec]], and BERT4Rec are effective but degrade under sparse and noisy interaction data. Prior contrastive-learning approaches improve robustness, yet most depend on hand-crafted stochastic augmentations at either the data level or model level, which makes them dataset-specific and often inefficient. The paper argues that many contrastive pairs are not informative enough to justify ever larger batch sizes or memory banks, and proposes to learn model augmentations adaptively so the system can generate more discriminative views from limited interaction data.

## Method

- **Task formulation**: given user sequence `S^u = {i_1^u, ..., i_|S^u|^u}`, predict the next interacted item with `arg max_{i in I} P(i^u_|S^u|+1 = i | S^u)`.
- **Backbone encoder**: item and positional embeddings form `e^u in R^(n x d)`, then a sequential encoder `f_theta(.)` produces `H^u = f_theta(e^u)` and uses the last hidden state `h_n^u` for prediction. In implementation, the encoder is a transformer-style [[sasrec]] backbone with `2` self-attention blocks, `2` heads, hidden size `d = 64`, and maximum sequence length `n = 50`.
- **Recommendation objective**: next-item prediction uses `softmax(h_n^u M^T)` with cross-entropy recommendation loss `L_rec`.
- **Stochastic data augmentation**: two augmentations are sampled from `G` to create `S~_1^u = g_1(S^u)` and `S~_2^u = g_2(S^u)`, where `g_1, g_2 ~ G` and `G` can include mask, crop, or reorder.
- **Learnable model augmentation**: the encoded views `h~^1` and `h~^2` are passed through two independent 3-layer [[multi-layer-perceptron]] augmenters, `z~^1 = w_phi1(h~^1)` and `z~^2 = w_phi2(h~^2)`, creating four total views for contrastive training.
- **Two contrastive losses**: `L_cl1 = L_con(h~^1, h~^2)` is standard data-view contrast; `L_cl2 = L_con(z~^1, z~^2) + L_con(h~^1, z~^2) + L_con(h~^2, z~^1)` couples data-augmented and model-augmented views. `L_con` is an InfoNCE-style loss over one positive pair and `2(|B|-1)` in-batch negatives.
- **Meta-optimized training**: stage 1 updates encoder parameters with `L_0 = L_rec + lambda L_cl1 + beta L_cl2 + gamma R`; stage 2 fixes the updated encoder and optimizes augmenter parameters with `L_1 = L_cl2 + gamma R`, iterating until convergence rather than jointly training all modules in one step.
- **Contrastive regularization**: similarities between `z~^1` and `z~^2` are split into positive set `sigma+` and negative set `sigma-`, then `R = 1/|sigma+| sum([sigma+ - o_min]_+) + 1/|sigma-| sum([o_max - sigma-]_+)` encourages positive pairs to stay close while keeping negatives separated.
- **Optimization details**: learning rates are `l = 0.001` and `l' = 0.001`, batch size is `256`, `lambda` and `beta` are tuned from `{0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}`, `gamma = 0.1 * beta`, and optimization uses Adam with early stopping on validation data.

## Key Results

- On **Sports**, MCLRec reaches `HR@20 = 0.0734` and `NDCG@20 = 0.0319`, improving over the best baseline by `5.46%` and `5.63%`, respectively.
- On **Beauty**, MCLRec reaches `HR@20 = 0.1243` and `NDCG@20 = 0.0539`, with relative gains of `4.19%` and `4.05%` over the best baseline.
- On **Yelp**, MCLRec reaches `HR@20 = 0.0941` and `NDCG@20 = 0.0467`, outperforming the best baseline by `8.41%` and `4.47%`.
- Across all metrics and datasets, the reported relative gains over the strongest baseline fall in the range `3.94%-8.41%` for HR and `2.47%-5.69%` for NDCG.
- Ablation on Sports shows removing model-augmentation contrastive loss drops performance from `HR@20/NDCG@20 = 0.0734/0.0319` to `0.0557/0.0238`, indicating that `L_cl2` contributes the largest share of the gains.
- Removing the regularizer reduces Yelp performance from `0.0941/0.0467` to `0.0873/0.0445`, and forcing the two augmenters to share parameters reduces Sports from `0.0734/0.0319` to `0.0707/0.0299`.

## Limitations

- Evaluation is limited to three public recommendation benchmarks (Sports, Beauty, Yelp); there is no online evaluation, no industrial-scale deployment result, and no evidence on domains outside sequential recommendation.
- The method still depends on hand-crafted stochastic data augmentation primitives (`mask`, `crop`, `reorder`) as the starting point, so the framework is not fully augmentation-free.
- Training is more complex than standard contrastive baselines because it alternates two optimization stages and introduces extra hyperparameters `lambda`, `beta`, and `gamma`; the paper reports dataset-specific best settings.
- The computational analysis is only coarse, with training dominated by `O(|U|^2 d)`; the paper does not provide wall-clock comparisons against all baselines under matched hardware budgets.
- The backbone encoder remains a SASRec-style transformer with modest scale (`d = 64`, `2` blocks, `2` heads), so it is unclear how well the gains transfer to stronger or more recent sequential recommenders.

## Concepts Extracted

- [[sequential-recommendation]]
- [[contrastive-learning]]
- [[data-augmentation]]
- [[model-augmentation]]
- [[meta-learning]]
- [[contrastive-loss]]
- [[representation-learning]]
- [[self-supervised-learning]]
- [[multi-layer-perceptron]]
- [[self-attention]]

## Entities Extracted

- [[xiuyuan-qin]]
- [[huanhuan-yuan]]
- [[pengpeng-zhao]]
- [[junhua-fang]]
- [[fuzhen-zhuang]]
- [[guanfeng-liu]]
- [[yanchi-liu]]
- [[victor-sheng]]
- [[beihang-university]]
- [[sasrec]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
