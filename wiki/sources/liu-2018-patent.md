---
type: source
subtype: paper
title: "Patent Litigation Prediction: A Convolutional Tensor Factorization Approach"
slug: liu-2018-patent
date: 2026-04-20
language: en
tags: [patent-mining, litigation-prediction, tensor-factorization, learning-to-rank, legal-analytics]
processed: true

raw_file: raw/papers/liu-2018-patent/paper.pdf
raw_md: raw/papers/liu-2018-patent/paper.md
bibtex_file: raw/papers/liu-2018-patent/paper.bib
possibly_outdated: true

authors:
  - Qi Liu
  - Han Wu
  - Yuyang Ye
  - Hongke Zhao
  - Chuanren Liu
  - Dongfang Du
year: 2018
venue: Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence
venue_type: conference
arxiv_id:
doi: 10.24963/ijcai.2018/701
url: https://www.ijcai.org/proceedings/2018/701
citation_key: liu2018patent
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2018; re-verify against recent literature.

The paper formulates patent litigation prediction as a ranking problem over triples of plaintiff company, defendant company, and patent. Its proposed Convolutional Tensor Factorization (CTF) combines a patent-content encoder, called NCNN, with a three-way tensor factorization over lawsuit histories so that sparse litigation records can be regularized by textual and meta-feature evidence from the patent itself. The NCNN branch encodes title, abstract, claims, and patent metadata, while the tensor branch models plaintiff-defendant, plaintiff-patent, and defendant-patent interactions. On a real-world USPTO plus Patexia dataset, the model consistently outperforms TF-only and content-only baselines, with more than 10% Precision@1 improvement over the second-best method in the leave-one-out setting.

## Problem & Motivation

Automatic early warning for patent litigation is difficult because lawsuit intent is heterogeneous, historical litigation records are extremely sparse relative to the full patent universe, and risky patents often depend on both relational context and patent content. The paper targets the concrete task: given a plaintiff-defendant company pair, rank patents by their likelihood of causing a lawsuit. The motivation is pragmatic risk management. If firms can identify risky patents before litigation occurs, they can adjust licensing, product, or legal strategy earlier and at lower cost.

## Method

- **Task formulation**: represent observed lawsuits as a 3D tensor `R`, where `r_ijk = 1` means plaintiff `i` sued defendant `j` over patent `k`; prediction targets unobserved triples `(i, j, k^-)`.
- **Ranking score**: compute patent risk by `r_hat_ijk = U_i^T V_j + U_i^T P_k + V_j^T P_k`, combining plaintiff-defendant, plaintiff-patent, and defendant-patent latent interactions.
- **Pairwise learning-to-rank**: for a litigated patent `k^+` and an unlitigated patent `k^-`, define `p(k^+ >_{i,j} k^-) = sigma(r_hat_ijk^+ - r_hat_ijk^-)`; the preference set is `D_R = {(i,j,k^+,k^-)|(i,j,k^+) in R and (i,j,k^-) not in R}`.
- **Patent-content bridge**: constrain the patent latent vector by `P_k = O_k + epsilon_k`, where `epsilon_k ~ N(0, delta_P^2 I)` and `O_k = NCNN(W, X_k)`, so content features regularize tensor factorization.
- **Text branch (NCNN)**: encode `C = 32` slices per patent (`title + abstract + first 30 claims`), truncate each slice to `H = 300` words, use pre-trained word embeddings of size `d_0`, and apply a two-layer CNN with word-level then sentence-level convolution and pooling.
- **Meta branch**: build a patent citation network and a `Q = 742` dimensional attribute vector per patent from citations, claims, pictures, sheets, CPC groups, grant lag, group trend, and assignee trend.
- **Attribute network embedding**: use DeepWalk-style sampled paths on the citation graph, define `e_k = E^T f_k`, aggregate neighbors by `e_context(k) = (1 / 2l) sum_j e_j`, and optimize the softmax objective with negative sampling to obtain a meta embedding.
- **Output layer**: concatenate text embedding and meta embedding, then pass them through a fully connected layer to produce `O_k`, the content-aware patent representation.
- **Optimization**: maximize the MAP objective over `U`, `V`, `P`, and `W`, equivalently minimize `-sum log sigma(r_hat_ijk^+ - r_hat_ijk^-) + lambda_U ||U||_F^2 + lambda_V ||V||_F^2 + lambda_W ||W||_F^2 + lambda_P ||P_k - NCNN(W, X_k)||_F^2`.
- **Hyperparameters**: latent dimension `|U_i| = |V_j| = |P_k| = 10`; initialization `N(0, 0.01I)`; `lambda_U = 10^-4`, `lambda_V = 10^-3`, `lambda_P = 10^-5`, `lambda_W = 10^-6`; learning rate `0.1`; optimization uses Adadelta with backpropagation in TensorFlow.

## Key Results

- Data scale: the raw corpus contains `6,422,962` granted US patents and `60,081` lawsuits from `2005-2016`; after filtering, experiments use `13,024` lawsuit pairs, `1,283` companies, `4,397` litigated patents, and `100,000` sampled non-litigated patents.
- Leave-one-out evaluation: CTF outperforms all baselines on `Precision@K`, `Recall@K`, and `F1@K`; the paper states that `Precision@1` improves by more than `10%` over the second-best method, Text-CTF.
- Percentage-wise evaluation: across training ratios from `20%` to `80%`, CTF achieves the best average `AUC`, indicating better robustness under sparse litigation histories.
- Ablation pattern: Meta-CTF and Text-CTF both beat TF, while full CTF beats LR and SVM, showing that both patent content and collaborative interaction structure matter.
- Case study 1: for MobileMedia Ideas vs Apple, the true litigated patent `5479476` receives `r_hat = 11.1058`, while a non-litigated candidate `6698825` receives `r_hat = -4.1640`.
- Case study 2: for Apple vs Oracle International, the litigated patent `5434872` scores `33.9545`, while a non-litigated candidate `6956752` scores `3.4850`.

## Limitations

- The evaluation is narrow: it uses one lawsuit dataset and one patent corpus, both US-centric, with preprocessing that removes companies having fewer than `2` litigation records.
- Negative examples are sampled rather than ranked against the full patent inventory, so practical deployment difficulty may be understated.
- The model depends on substantial feature engineering, including `742` handcrafted metadata dimensions and manual truncation to `30` claims and `300` words per slice.
- Main-text reporting emphasizes relative wins and plots, but does not provide a full table of exact `Precision@K` / `Recall@K` / `F1@K` values in the paper body.
- The method is specialized to lawsuit-history tensors and does not test transfer to broader legal prediction or patent-analysis tasks.

## Concepts Extracted

- [[patent-litigation-prediction]]
- [[convolutional-tensor-factorization]]
- [[tensor-factorization]]
- [[collaborative-filtering]]
- [[learning-to-rank]]
- [[network-embedding]]
- [[attribute-network-embedding]]
- [[convolutional-neural-network]]
- [[bayesian-personalized-ranking]]
- [[pairwise-ranking-loss]]
- [[leave-one-out-evaluation]]

## Entities Extracted

- [[qi-liu]]
- [[han-wu]]
- [[yuyang-ye]]
- [[hongke-zhao]]
- [[chuanren-liu]]
- [[dongfang-du]]
- [[university-of-science-and-technology-of-china]]
- [[drexel-university]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
