---
type: source
subtype: paper
title: Disentangled Contrastive Collaborative Filtering
slug: ren-2023-disentangled
date: 2026-04-20
language: en
tags: [recommendation, collaborative-filtering, graph-learning, contrastive-learning, disentanglement]
processed: true

raw_file: raw/papers/ren-2023-disentangled/paper.pdf
raw_md: raw/papers/ren-2023-disentangled/paper.md
bibtex_file: raw/papers/ren-2023-disentangled/paper.bib
possibly_outdated: true

authors:
  - Xubin Ren
  - Lianghao Xia
  - Jiashu Zhao
  - Dawei Yin
  - Chao Huang
year: 2023
venue: SIGIR 2023
venue_type: conference
arxiv_id:
doi: 10.1145/3539618.3591665
url: https://dl.acm.org/doi/10.1145/3539618.3591665
citation_key: ren2023disentangled
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

DCCF is a graph-based collaborative filtering model that combines disentangled intent modeling with adaptive contrastive augmentation. The paper argues that prior graph contrastive recommenders under-model multiple latent user intents and rely on fixed augmentations that can amplify noisy interactions. DCCF addresses this by learning user/item intent prototypes, injecting global intent-aware context into message passing, and constructing learnable edge masks for augmentation rather than using random dropouts. The model is trained with a multitask objective that combines Bayesian Personalized Ranking with three InfoNCE-style contrastive alignments across original, disentangled-global, and adaptively augmented views. On Gowalla, Amazon-book, and Tmall, DCCF outperforms strong GNN, disentanglement, and self-supervised recommendation baselines while also improving robustness to sparsity and over-smoothing.

## Problem & Motivation

Existing GNN-based collaborative filtering models benefit from high-order propagation, but they depend on sparse implicit-feedback labels. Graph contrastive learning helps by adding self-supervision, yet the paper identifies two gaps. First, most methods learn entangled user/item embeddings and therefore blur multiple latent intent factors behind interactions. Second, common augmentation schemes such as random edge or node dropout are non-adaptive and may inject misleading self-supervised signals when the interaction graph contains noise, popularity bias, or misclicks. The paper targets a recommender that can disentangle intent structure and learn augmentation weights from the graph itself, so self-supervision is both finer-grained and more robust.

## Method

- **Interaction modeling**: define the user-item matrix `A in R^(I x J)` and decompose preference prediction through latent user/item intents, approximating `E_{P(c_u|u)P(c_v|v)}[f(c_u, c_v)]` with `f(E[c_u], E[c_v])` to obtain tractable intent-aware scoring.
- **Base propagation**: use normalized bipartite message passing `Z^(u) = A_bar E^(v)` and `Z^(v) = A_bar^T E^(u)`, then residual updates `E_l = E_(l-1) + Z_(l-1)` to capture high-order collaborative signals.
- **Disentangled global context**: introduce `K` learnable intent prototypes for users and items, compute soft intent assignment with `P(c_u^k | e_i) = exp(e_i^T c_u^k) / sum_k' exp(e_i^T c_u^k')`, and aggregate global intent-aware embeddings `r_i = sum_k c_u^k P(c_u^k | e_i)` and `r_j = sum_k c_v^k P(c_v^k | e_j)`.
- **Intent-aware refinement**: fuse local and global signals via `E_l^(u) = E_(l-1)^(u) + Z_(l-1)^(u) + R_(l-1)^(u)` and the analogous item update, so each layer combines neighborhood evidence with disentangled global dependencies.
- **Adaptive augmentation**: for each layer learn a mask `M_ij^l = (s(r_i, r_j) + 1) / 2` from cosine similarity, build a masked relation matrix `G^l = M^l odot A`, normalize it, and generate augmented propagation views `H^(u) = G_bar E^(v)` and `H^(v) = G_bar^T E^(u)`. A second mask over local embeddings gives another adaptive view.
- **Contrastive objective**: align original embeddings `z` with three view types using InfoNCE, namely the disentangled global view `r`, local adaptive view `h^beta`, and global adaptive view `h^gamma`, then aggregate them as `L_cl^(u) = I(z, r) + I(z, h^beta) + I(z, h^gamma)` plus the symmetric item-side loss.
- **Training objective and hyperparameters**: optimize `L = L_bpr + lambda_1 (L_cl^(u) + L_cl^(v)) + lambda_2 ||Theta_1||_F^2 + lambda_3 ||Theta_2||_F^2`, with `L_bpr = -(1/|R|) sum log sigma(Y_i,p - Y_i,n)`. Implementation uses PyTorch, Adam, learning rate `1e-3`, embedding size `d = 32`, batch size `10240`, propagation depth chosen from `{1, 2, 3}`, and default intent count `K = 128`.
- **Complexity**: the paper states message passing costs `O(L |A| d)`, intent aggregation costs `O(L (I + J) K d)`, two adaptive augmenters cost `O(2 L |A| d)`, and the contrastive term costs `O(L B (I + J) d)`.

## Key Results

- **Overall accuracy**: DCCF is best on all three datasets. On Gowalla it reaches `Recall@20 = 0.1876` and `NDCG@20 = 0.1123`, beating LightGCL (`0.1825`, `0.1077`) and HCCF (`0.1818`, `0.1061`).
- **Amazon-book**: DCCF achieves `Recall@20 = 0.0889`, `Recall@40 = 0.1343`, `NDCG@20 = 0.0680`, `NDCG@40 = 0.0829`, exceeding the strongest baseline in each column with reported `p`-values down to `8.6e-7`.
- **Tmall**: DCCF reaches `Recall@20 = 0.0668` and `NDCG@20 = 0.0469`, outperforming LightGCL (`0.0632`, `0.0444`) and HCCF (`0.0623`, `0.0425`).
- **Ablation evidence**: removing disentangled encoding drops Gowalla `Recall@20` from `0.1876` to `0.1637`; removing all adaptive augmented views reduces Amazon-book `NDCG@20` from `0.0680` to `0.0632`.
- **Over-smoothing robustness**: DCCF attains higher MAD than DCCF-CL on Amazon-book user embeddings (`0.999` vs `0.902`) and Tmall user embeddings (`0.999` vs `0.800`), indicating less severe over-smoothing.
- **Efficiency**: per-epoch training time is `12.4s` on Gowalla, `18.9s` on Amazon-book, and `18.8s` on Tmall, faster than DGCF (`25.1s`, `49.6s`, `51.6s`) though still slower than DGCL (`9.3s`, `12.4s`, `12.0s`).
- **Intent count study**: performance improves as `K` increases from `32` to `128`, but gains saturate or degrade at `K = 256`, especially on Tmall, suggesting overly fine-grained intent prototypes introduce redundancy.

## Limitations

- The evaluation is limited to three implicit-feedback recommendation datasets; there is no evidence for broader generalization to other domains or explicit-feedback settings.
- The gains over the strongest SSL baselines are consistent but sometimes modest, so the extra machinery of intent prototypes and learned masks must justify its added complexity.
- Adaptive masking still depends on observed interaction structure; if the graph is systematically biased, the learned augmenters may inherit that bias rather than remove it.
- The method introduces several coupled hyperparameters, including `K`, `lambda_1`, `lambda_2`, `lambda_3`, and layer count, which can complicate tuning and reproducibility.
- The paper does not compare against more recent post-2023 recommendation or graph SSL methods, so the claimed frontier status is time-bounded.

## Concepts Extracted

- [[collaborative-filtering]]
- [[graph-neural-network]]
- [[graph-contrastive-learning]]
- [[disentangled-representation]]
- [[latent-intent]]
- [[adaptive-data-augmentation]]
- [[infonce-loss]]
- [[bayesian-personalized-ranking]]
- [[data-sparsity]]
- [[over-smoothing]]

## Entities Extracted

- [[xubin-ren]]
- [[lianghao-xia]]
- [[jiashu-zhao]]
- [[dawei-yin]]
- [[chao-huang]]
- [[university-of-hong-kong]]
- [[wilfrid-laurier-university]]
- [[baidu]]
- [[gowalla]]
- [[amazon-book]]
- [[tmall]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
