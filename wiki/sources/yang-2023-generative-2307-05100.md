---
type: source
subtype: paper
title: Generative Contrastive Graph Learning for Recommendation
slug: yang-2023-generative-2307-05100
date: 2026-04-20
language: en
tags: [recommendation, collaborative-filtering, graph-learning, contrastive-learning, variational-inference]
processed: true

raw_file: raw/papers/yang-2023-generative-2307-05100/paper.pdf
raw_md: raw/papers/yang-2023-generative-2307-05100/paper.md
bibtex_file: raw/papers/yang-2023-generative-2307-05100/paper.bib
possibly_outdated: true

authors:
  - Yonghui Yang
  - Zhengwei Wu
  - Le Wu
  - Kun Zhang
  - Richang Hong
  - Zhiqiang Zhang
  - Jun Zhou
  - Meng Wang
year: 2023
venue: SIGIR 2023
venue_type: conference
arxiv_id: 2307.05100
doi: 10.1145/3539618.3591691
url: http://arxiv.org/abs/2307.05100
citation_key: yang2023generative
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

VGCL targets collaborative filtering with graph contrastive learning but removes the usual dependence on manually designed structure or feature augmentations. Starting from a user-item bipartite graph, it uses LightGCN-based variational inference to estimate a Gaussian posterior `q(z_i|A,E^0)=N(\mu_i, diag(\sigma_i^2))` for each node, reconstructs the graph with an inner-product decoder, and samples multiple stochastic views from the learned distribution for contrastive training. The contrastive objective is twofold: node-level consistency across samples of the same node and cluster-level consistency across nodes assigned to the same prototype by K-means. On Douban-Book, Dianping, and MovieLens-25M, the method consistently beats LightGCN, SGL, NCL, SimGCL, and VAE baselines, suggesting that learned generative views preserve recommendation-relevant graph structure better than hand-crafted augmentation.

## Problem & Motivation

The paper studies graph-based collaborative filtering under sparse supervision. Prior graph contrastive learning methods improve recommendation quality by creating two augmented graph views, but the authors argue that both mainstream augmentation families are flawed in this setting: random edge/node dropout can damage the intrinsic dependency structure of the user-item graph, while fixed-scale feature noise treats all nodes identically despite large differences in interaction density. The core motivation is to generate contrastive views that remain faithful to the original graph while adapting the perturbation scale to each user or item.

## Method

- **Base graph formulation**: recommendation data are modeled as a bipartite graph `G = {U \cup V, A}` with adjacency matrix `` `A = [[0, R], [R^T, 0]]` ``, where `R` is the implicit-feedback interaction matrix.
- **Graph encoder**: the framework follows LightGCN-style propagation, `` `E^l = D^{-1/2} A D^{-1/2} E^{l-1}` ``, to exploit higher-order collaborative signals without heavy graph transformations.
- **Variational graph inference**: each node `i` is assigned a Gaussian posterior `` `q_\phi(z_i|A,E^0) = N(\mu_i, diag(\sigma_i^2))` ``. Means are aggregated from graph layers and fused as `` `\mu = (1/L) \sum_{l=1}^L \mu^l` ``, while variances come from a one-layer MLP, `` `\sigma = exp(\mu W + b)` ``.
- **Reparameterized sampling**: latent views are drawn with `` `z_i = \mu_i + \sigma_i \cdot \varepsilon` `` and two contrastive instances per node are formed as `` `z_i' = \mu_i + \sigma_i \cdot \varepsilon'` `` and `` `z_i'' = \mu_i + \sigma_i \cdot \varepsilon''` ``, where noise is Gaussian.
- **Graph reconstruction objective**: the decoder uses an inner product, `` `p(A_ij=1|z_i,z_j) = sigmoid(z_i^T z_j)` ``, and optimizes an ELBO-style reconstruction term with pairwise training triples similar to BPR.
- **Twofold contrastive learning**: node-level losses `` `L_N^U` `` and `` `L_N^I` `` use InfoNCE with temperature `` `\tau_1` `` over users and items separately. Cluster-level losses `` `L_C^U` `` and `` `L_C^I` `` weight positives by shared-cluster probability computed from K-means prototypes `` `C^u \in R^{d \times K_u}` `` and `` `C^i \in R^{d \times K_i}` ``.
- **Final optimization**: the full objective is `` `L = L_ELBO + \alpha L_cl + \lambda ||E^0||^2` `` with `` `L_cl = L_N + \gamma L_C` ``. This explicitly couples generative reconstruction and contrastive regularization.
- **Implementation / hyperparameters**: embedding size is fixed to `` `d = 64` ``, optimizer is Adam with learning rate `` `1e-3` ``, batch size is `` `2048` `` on Douban-Book and Dianping and `` `4096` `` on MovieLens-25M. The paper tunes `` `\tau \in [0.10, 0.25]` ``, `` `\lambda \in [0.01, 1.0]` ``, and `` `K_u, K_i \in [100, 1000]` ``; best reported `` `\gamma` `` values are `0.4` (Douban-Book), `0.5` (Dianping), and `1.0` (MovieLens-25M).
- **Efficiency claim**: because VGCL runs graph convolution once and samples views in latent space, the paper argues graph propagation cost is `` `O(2|E|dS)` `` versus `` `O(6|E|dS)` `` for augmentation-based methods such as SGL and SimGCL that require three encoder passes.

## Key Results

- On **Douban-Book**, VGCL reaches `Recall@20 = 0.1829` and `NDCG@20 = 0.1638`, beating LightGCN (`0.1516`, `0.1278`) and SimGCL (`0.1731`, `0.1540`).
- On **Dianping**, VGCL achieves `Recall@20 = 0.1234` and `NDCG@20 = 0.0757`, compared with LightGCN (`0.1076`, `0.0660`) and SimGCL (`0.1208`, `0.0743`).
- On **MovieLens-25M**, VGCL obtains `Recall@20 = 0.3507` and `NDCG@20 = 0.2725`, again outperforming LightGCN (`0.3263`, `0.2509`) and SimGCL (`0.3491`, `0.2690`).
- Relative to LightGCN, the paper reports NDCG@20 gains of `28.17%`, `14.70%`, and `8.61%` on Douban-Book, Dianping, and MovieLens-25M respectively.
- Ablations show both main components matter: removing the cluster-level loss gives Douban-Book `NDCG@20 = 0.1575`, while removing variational reconstruction gives `0.1547`; the full model reaches `0.1638`.
- Hyperparameter analysis shows the best graph inference depth is shallow (`L = 2` on Douban-Book, `L = 3` on Dianping), and the best cluster-level temperature is lower than SimGCL's typical `0.2` (`0.13` on Douban-Book, `0.15` on Dianping).
- User-group analysis reports larger gains for denser users, with VGCL improving over SimGCL by `8.4%` in the densest Douban-Book group versus `4.2%` in the sparsest group, consistent with the claim that learned variances adapt augmentation scale to node density.

## Limitations

- The evaluation is limited to three offline recommendation benchmarks and top-N metrics; there is no online serving study, cold-start analysis, or robustness check under distribution shift.
- The method introduces several sensitive hyperparameters, including `` `K_u` ``, `` `K_i` ``, `` `\tau_2` ``, `` `\alpha` ``, and `` `\gamma` ``, and the best settings differ noticeably across datasets.
- The variational module assumes a Gaussian latent distribution with an inner-product graph decoder, which may be too restrictive for richer interaction semantics or side-information-heavy recommendation settings.
- The cluster-level objective depends on K-means prototypes; although the paper claims Faiss-GPU makes clustering cheap, it does not provide a separate systems benchmark for very large catalogs.
- The baseline set is strong for 2023 but does not test against later recommendation architectures, so the margin should be re-verified against newer literature.

## Concepts Extracted

- [[collaborative-filtering]]
- [[graph-contrastive-learning]]
- [[contrastive-learning]]
- [[graph-neural-network]]
- [[variational-inference]]
- [[variational-autoencoder]]
- [[variational-graph-reconstruction]]
- [[cluster-aware-contrastive-learning]]
- [[k-means-clustering]]
- [[pairwise-ranking-loss]]
- [[lightgcn]]
- [[data-augmentation]]

## Entities Extracted

- [[yonghui-yang]]
- [[zhengwei-wu]]
- [[le-wu]]
- [[kun-zhang-hfut]]
- [[richang-hong]]
- [[zhiqiang-zhang]]
- [[jun-zhou]]
- [[meng-wang-hfut]]
- [[ant-group]]
- [[hefei-university-of-technology]]
- [[douban-book]]
- [[dianping]]
- [[movielens-25m]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
