---
type: source
subtype: paper
title: Multi-Scale Subgraph Contrastive Learning
slug: liu-2023-multiscale
date: 2026-04-20
language: en
tags: [graph-learning, contrastive-learning, self-supervised-learning, graph-neural-network, graph-classification]
processed: true

raw_file: raw/papers/liu-2023-multiscale/paper.pdf
raw_md: raw/papers/liu-2023-multiscale/paper.md
bibtex_file: raw/papers/liu-2023-multiscale/paper.bib
possibly_outdated: false

authors:
  - Yanbei Liu
  - Yu Zhao
  - Xiao Wang
  - Lei Geng
  - Zhitao Xiao
year: 2023
venue: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence
venue_type: conference
arxiv_id:
doi: 10.24963/ijcai.2023/246
url: https://www.ijcai.org/proceedings/2023/246
citation_key: liu2023multiscale
paper_type: method

read_status: unread

domain: graph-learning
---

## Summary

This paper argues that graph contrastive learning should not treat all augmented subgraphs from the same graph as uniformly positive, because semantic similarity depends strongly on subgraph scale. The authors first show on four TUDataset benchmarks that larger sampled subgraphs have higher mean semantic similarity and lower variance than smaller ones. They then propose MSSGCL, which constructs global and local subgraph views and optimizes three relations jointly: global-global attraction, global-local attraction, and local-local repulsion mediated by a learned regressor. Across eight graph classification datasets, the method consistently improves over prior graph contrastive learning baselines in both unsupervised and semi-supervised settings.

## Problem & Motivation

Existing graph contrastive learning methods usually assume that two augmented views from the same graph should always be pulled together as a positive pair. The paper argues that this assumption is too coarse because graph semantics are multi-scale: small subgraphs may capture quite different local content, while larger subgraphs preserve more of the original graph's global semantics. If a method enforces similarity between semantically mismatched local augmentations, the resulting supervision becomes noisy or contradictory. The motivation for MSSGCL is therefore to model semantic associations at multiple scales and assign different contrastive objectives to global and local subgraph pairs.

## Method

- **Multi-scale augmentation**: for each input graph `G`, the model samples two global views `{\hat{G}_i^g}_{i=1}^2` and two local views `{\hat{G}_i^l}_{i=1}^2` using random-walk-based `subgraph sampling`, with separate augmentation distributions `T_g(.|G)` and `T_l(.|G)`.
- **Scale control**: in experiments, molecular graphs use global/local view sizes of `80%` and `20%` of the original graph; social-network graphs use `90%` and `10%`, respectively.
- **Encoder and projector**: a `K`-layer GNN encoder computes node states via `a_n^(k) = AGGREGATE^(k)({h_u^(k-1) : u in N(n)})` and `h_n^(k) = COMBINE^(k)(h_n^(k-1), a_n^(k))`; a readout produces graph embedding `f(\hat{G})`, then a non-linear projector maps it to `z = g(f(\hat{G}))`.
- **Global-global loss**: global views from the same graph are optimized with an InfoNCE-style objective `l_s(z_1^g, z_2^g) = -log(exp(z_1^g · z_2^g / tau) / (exp(z_1^g · z_2^g / tau) + sum exp(z_1^g · z_-^g / tau)))`, aggregated as `L_gg = E[l_s(z_1^g, z_2^g)]`.
- **Global-local loss**: the model also aligns global and local views via `L_gl = E[sum_{i=1}^2 (l_s(z_i^g, z_1^l) + l_s(z_i^g, z_2^l))]`, reflecting that global views should partially contain local semantics.
- **Local-local loss**: instead of directly maximizing local-view similarity, MSSGCL assumes local views may describe different content and uses a learnable regressor `f_{theta_d}: R^n x R^n -> R^+` trained with `psi(z_1^l, z_2^l) = E[f_{theta_d}(z_1^l, z_2^l)] - E[f_{theta_d}(z_1^l, z_-^l)]`; the encoder then minimizes `L_ll = E[l_d(z_1^l, z_2^l)]`.
- **Total objective**: training solves `min L_gg + lambda_1 L_gl + lambda_2 L_ll` with bi-level updates over the encoder and the local-view regressor.
- **Implementation details**: the local-view similarity regressor is a `5`-layer MLP with batch normalization, ReLU activations, and a final Sigmoid output; unsupervised experiments use `GIN` plus sum pooling, while semi-supervised experiments use a `5`-layer `ResGCN` with hidden size `128`.

## Key Results

- On unsupervised graph classification, MSSGCL achieves the best average accuracy across eight datasets: `77.52`, outperforming `SimGRACE` (`76.17`) and `GraphCL` (`75.41`).
- Unsupervised per-dataset results include `81.45 +- 0.48` on `NCI1`, `75.49 +- 0.70` on `PROTEINS`, `79.73 +- 0.44` on `DD`, `89.68 +- 0.57` on `MUTAG`, `73.48 +- 0.83` on `COLLAB`, `91.08 +- 0.78` on `RDT-B`, `56.17 +- 0.18` on `RDT-M5K`, and `73.14 +- 0.38` on `IMDB-B`.
- With `1%` labels in the semi-supervised setting, MSSGCL reaches average accuracy `64.88`, beating `SimGRACE` (`64.25`) and `GraphCL` (`63.56`).
- With `10%` labels, MSSGCL reaches average accuracy `75.06` and is best on `6/7` reported datasets, including `75.76` on `PROTEINS`, `78.89` on `DD`, `76.02` on `COLLAB`, `90.58` on `RDT-B`, and `54.36` on `RDT-M5K`.
- Relative to `GraphCL`, the paper reports roughly `2` percentage points average improvement in the `10%`-label semi-supervised setting.
- Ablations show each loss term matters: removing `L_gg` drops performance to `80.27` on `NCI1`, `78.62` on `DD`, `71.92` on `COLLAB`, and `88.87` on `RDT-B`, while removing `L_ll` also causes consistent degradation.

## Limitations

- The empirical study is restricted to graph classification benchmarks; the paper does not test transfer to other graph tasks such as link prediction or node classification.
- The semantic-similarity analysis depends on a supervised `5`-layer GIN trained on four TUDataset datasets, so the diagnostic itself inherits assumptions from that encoder and benchmark choice.
- The method uses fixed heuristic view-size ratios (`80/20` or `90/10`) rather than learning scale selection adaptively per graph.
- The local-local objective assumes local views from the same graph should often remain separated; this may fail for graphs whose small subgraphs are genuinely redundant or highly repetitive.
- The paper reports accuracy gains but provides little discussion of computational overhead from maintaining both multi-scale views and an extra learned regressor.

## Concepts Extracted

- [[graph-contrastive-learning]]
- [[contrastive-learning]]
- [[self-supervised-learning]]
- [[graph-neural-network]]
- [[graph-augmentation]]
- [[subgraph-sampling]]
- [[graph-representation-learning]]
- [[graph-classification]]
- [[message-passing]]
- [[graph-isomorphism-network]]

## Entities Extracted

- [[yanbei-liu]]
- [[yu-zhao]]
- [[xiao-wang]]
- [[lei-geng]]
- [[zhitao-xiao]]
- [[tiangong-university]]
- [[beihang-university]]
- [[tudataset]]
- [[mutag-dataset]]
- [[nci1-dataset]]
- [[dd-dataset]]
- [[proteins-dataset]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
