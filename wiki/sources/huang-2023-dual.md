---
type: source
subtype: paper
title: Dual Contrastive Transformer for Hierarchical Preference Modeling in Sequential Recommendation
slug: huang-2023-dual
date: 2026-04-20
language: en
tags: [sequential-recommendation, recommender-systems, contrastive-learning, transformer, knowledge-graph]
processed: true
raw_file: raw/papers/huang-2023-dual/paper.pdf
raw_md: raw/papers/huang-2023-dual/paper.md
bibtex_file: raw/papers/huang-2023-dual/paper.bib
possibly_outdated: true
authors:
  - Chengkai Huang
  - Shoujin Wang
  - Xianzhi Wang
  - Lina Yao
year: 2023
venue: SIGIR 2023
venue_type: conference
arxiv_id:
doi: 10.1145/3539618.3591672
url: https://dl.acm.org/doi/10.1145/3539618.3591672
citation_key: huang2023dual
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper proposes HPM, a sequential recommendation framework that explicitly separates user preference dynamics into low-level item preference and higher-level category preference. It combines a dual-transformer backbone over item and category sequences, semantics-enhanced target embeddings derived from relation-aware TransE signals, and dual contrastive objectives that align both item-level and category-level user representations with the next target. The method is evaluated on six Amazon 5-core subsets with sequence length capped at `20`, embedding size `64`, one transformer layer, and `4` attention heads. Across all datasets and all reported HR/NDCG cutoffs, HPM outperforms strong temporal, category-aware, and contrastive baselines, with relative gains over the best baseline ranging from `1.53%` to `17.61%`.

## Problem & Motivation

Existing sequential recommender systems usually model only item-ID-level preference shifts, even though user behavior also exhibits slower-moving category-level preference dynamics. Prior contrastive sequential recommenders mostly contrast augmented item sequences and therefore miss category-level supervision while sometimes corrupting temporal structure. The paper also argues that short and sparse interaction sequences make standard context embeddings under-informative because they ignore explicit semantic relations such as `also_buy`, `also_view`, same-brand, and same-category-with-similar-price. HPM is designed to address these three gaps jointly: hierarchical preference modeling, relation-aware context enhancement, and non-destructive dual-level contrastive learning.

## Method

- **Input representation**: each user context is `O_i = {V_i, C_i, T_i}`, where `V_i` is the item-ID sequence, `C_i` the category sequence, and `T_i` timestamps; the task is next-item prediction from the first `n-1` interactions.
- **Embedding layer**: learn item embeddings `E_V in R^{|V| x d}`, category embeddings `E_C in R^{|C| x d}`, and position embeddings `E_P in R^{L x d}` with position-sensitive inputs `e_{i,v} = e_{i,v} + p_i` and `e_{i,c} = e_{i,c} + p_i`.
- **Relation pretraining**: use TransE on item and category relation triples with `f(v_h, r, v_t) = ||e_{v,h} + e_r - e_{v,t}||_2^2` and the analogous category loss to encode explicit semantic relations.
- **Dual Transformer (DT)**: run two self-attention transformers in parallel, one over item IDs and one over categories, using `Attention(Q, K, V) = softmax(QK^T / sqrt(d / h)) V`, FFN, residual connections, dropout, and layer normalization.
- **Hierarchical preference readout**: obtain low- and high-level user states via average pooling, `v_f = (1 / L) sum_l S_{i,v_l}` and `c_f = (1 / L) sum_l S_{i,c_l}`.
- **SCEL module**: enhance target embeddings with temporal relation signals. Complementary relations (`also_buy`) use Gaussian decay `phi^1(Delta t) = N(Delta t | 0, sigma)`, while substitute relations (`also_view`) use a short-term negative plus long-term positive kernel `phi^2(Delta t) = -N(Delta t | 0, sigma) + N(Delta t | mu, sigma)`.
- **Dual contrastive learning (DCL)**: align item-level and category-level user representations with semantics-enhanced target embeddings using separate InfoNCE-style losses `L_cl_item` and `L_cl_cate`, then combine them as `L_cl = L_cl_item + L_cl_cate`.
- **Recommendation objective**: optimize BPR with scores `y_hat_ui = e_{v,n}^T v_{f,i} + e_{c,n}^T c_{f,i}` and total loss `L_joint = L_rec + lambda L_cl`.
- **Training details**: maximum sequence length `20`, embedding size `64`, batch size `64`, negative samples `1`, one self-attention layer, `4` heads, `lambda = 1`, Adam learning rate `1e-5` for knowledge embedding pretraining and `1e-6` for the main model, early stopping after `10` non-improving validation rounds, and up to `200` epochs on a single NVIDIA TITAN RTX 24 GB GPU.

## Key Results

- HPM beats the strongest baseline on every dataset and every reported metric, with relative gains ranging from `1.53%` to `17.61%`; reported improvements are significant at `p < 0.05`.
- On Clothing, the model reaches `HR@10 = 0.5748` and `NDCG@5 = 0.3387`, improving over the best baseline by `15.17%` and `17.61%`, respectively.
- On Cellphones, HPM obtains `HR@20 = 0.8225`, `HR@50 = 0.9428`, and `NDCG@10 = 0.4882`, corresponding to gains of `3.81%`, `1.78%`, and `7.94%`.
- On Grocery, it reports `HR@5 = 0.5432`, `HR@20 = 0.7514`, and `NDCG@50 = 0.4985`, beating the best baseline by `5.11%`, `1.53%`, and `3.04%`.
- On Beauty, HPM achieves `HR@5 = 0.5141`, `HR@20 = 0.7424`, and `NDCG@10 = 0.4239`, with gains of `4.78%`, `2.81%`, and `4.93%`.
- Ablations show all three modules matter: removing SCEL yields the worst variant across tested datasets, replacing the dual transformer with a single fused transformer degrades performance, and dropping DCL consistently hurts HR@5 and NDCG@5.

## Limitations

- Evaluation is restricted to six Amazon sub-datasets under a `5-core` filtering protocol and sampled ranking with `99` negatives, so external validity to denser or non-e-commerce domains is unclear.
- The hierarchical signal is limited to item categories; richer side information such as price, brand, text, or multimodal product content is not modeled directly.
- SCEL depends on hand-specified relation types (`also_buy`, `also_view`, same brand, similar-price same-category) and TransE pretraining quality, which may be noisy or unavailable in other datasets.
- The paper reports single-GPU experiments with `d = 64` and shallow transformers; it does not study scalability to longer sequences or larger architectures.
- Contrastive learning uses in-batch negatives only; the paper does not analyze false negatives or calibration effects induced by sampled negatives in recommendation.

## Concepts Extracted

- [[sequential-recommendation]]
- [[hierarchical-preference-modeling]]
- [[dual-transformer]]
- [[semantics-enhanced-context-embedding-learning]]
- [[dual-contrastive-learning]]
- [[contrastive-learning]]
- [[category-aware-recommendation]]
- [[knowledge-graph-embedding]]
- [[bayesian-personalized-ranking]]
- [[self-attention]]
- [[multi-head-attention]]

## Entities Extracted

- [[chengkai-huang]]
- [[shoujin-wang]]
- [[xianzhi-wang]]
- [[lina-yao]]
- [[university-of-new-south-wales]]
- [[university-of-technology-sydney]]
- [[csiro-data61]]
- [[sasrec]]
- [[transe]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
