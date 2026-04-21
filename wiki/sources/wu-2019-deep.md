---
type: source
subtype: paper
title: Deep Technology Tracing for High-Tech Companies
slug: wu-2019-deep
date: 2026-04-20
language: en
tags: [patent-mining, technology-forecasting, neural-ranking, temporal-modeling, company-analysis]
processed: true

raw_file: raw/papers/wu-2019-deep/paper.pdf
raw_md: raw/papers/wu-2019-deep/paper.md
bibtex_file: raw/papers/wu-2019-deep/paper.bib
possibly_outdated: false

authors:
  - Han Wu
  - Kun Zhang
  - Guangyi Lv
  - Qi Liu
  - Runlong Yu
  - Weihao Zhao
  - Enhong Chen
  - Jianhui Ma
year: 2019
venue: 2019 IEEE International Conference on Data Mining (ICDM)
venue_type: conference
arxiv_id:
doi: 10.1109/ICDM.2019.00180
url: https://ieeexplore.ieee.org/document/8970847/
citation_key: wu2019deep
paper_type: method

read_status: unread
read_date:
rating:

domain: patent-mining
---

## Summary

This paper studies firm-specific technology forecasting from patent histories and proposes Deep Technology Forecasting (DTF), a three-part framework that combines Potential Competitor Recognition (PCR), Collaborative Technology Recognition (CTR), and a Deep Technology Tracing (DTT) neural network. The core idea is to forecast next-year company-level technology distributions rather than market-wide technology trends, using both internal firm patent content and external signals from competitors and related technologies. DTT encodes patent text with CNNs, injects relation-weighted competitor and technology signals, models yearly dynamics with GRUs, and predicts company-technology affinity with a BPR-trained scorer. On large-scale USPTO data, the paper reports consistent gains over tensor-factorization, linear, and indicator-based baselines, plus qualitatively plausible top-10 technology forecasts in a Hughes Network Systems case study.

## Problem & Motivation

The paper targets technology tracing for high-tech companies: given a company's historical patents, predict which technologies it will emphasize next year. The authors argue that prior technology forecasting methods are either market-level or indicator-driven, so they miss firm-specific demand, competitive pressure, technological co-evolution, and temporal dynamics. Their formulation therefore combines internal factors (a firm's own patent content and portfolio), external factors (technology evolution in the market), company-company competition, technology-technology collaboration, and time-varying company-technology interactions.

## Method

- **Prediction target**: for company `U_i` and technology `V_j`, the yearly technology share is `r_{i,j}^t = |S_{U_i^t} ∩ S_{V_j^t}| / |S_{U_i^t}|`, and the model forecasts the next-year distribution vector `r_i^T`.
- **Potential Competitor Recognition (PCR)**: compute competition with three patent indicators, including patent activity `I_1`, technology share `I_2 = |S_{U_i^t} ∩ S_{V_j^t}| / |S_{V_j^t}|`, and R&D emphasis `I_3 = |S_{U_i^t} ∩ S_{V_j^t}| / |S_{U_i^t}|`; competitor distance is `pcr^t(U_{i_1}, U_{i_2}) = sqrt(sum_q alpha_q (I_q^{U_{i_1},t} - I_q^{U_{i_2},t})^2)`, then top-`m` competitors are selected.
- **Collaborative Technology Recognition (CTR)**: build yearly patent-technology bipartite graphs, then define collaboration between technologies with a Jaccard-style overlap `ctr^t(V_{j_1}, V_{j_2}) = |S_{V_{j_1}^t} ∩ S_{V_{j_2}^t}| / |S_{V_{j_1}^t} ∪ S_{V_{j_2}^t}|`, retaining top-`n` collaborators.
- **Patent content encoder**: each patent is a word sequence `e = [e_1, ..., e_{d_1}]` with pre-trained embeddings `e_i in R^{d_0}`; a three-stage CNN + pooling stack maps each patent to `e_dot in R^d`.
- **Relation-enhanced factors**: each company/year samples `d_2` patents; internal factor tensor is `D_i in R^{(m+1)*d_2*d_1*d_0}` and external factor tensor is `D_j in R^{(n+1)*d_2*d_1*d_0}`. Aggregated yearly embeddings are `x_i^t = a_i^t + sum_{i' in PC_i^t} pcr^t(U_i, U_{i'}) * a_{i'}^t` and `y_j^t = a_j^t + sum_{j' in CT_j^t} ctr^t(V_j, V_{j'}) * a_{j'}^t`.
- **Temporal dynamics**: GRUs consume yearly company embeddings `x_i^1 ... x_i^{T-1}` and technology embeddings `y_j^1 ... y_j^{T-1}` to produce dynamic states `u_i^t` and `v_j^t`; the paper writes the standard GRU update/reset equations in Eq. (6).
- **Scoring and training**: next-step relevance is `r_hat_uv = sigma(u . v)`. Training uses Bayesian Personalized Ranking with `L = sum_{(i,j+,j-) in D_S} -ln sigma(r_hat_{ij+} - r_hat_{ij-}) + lambda ||Theta||^2`, optimized with Adadelta in TensorFlow.

## Key Results

- The benchmark covers `6,014,932` USPTO patents (1972-2017), `389,246` assignees, `2,791` high-tech companies after filtering, and `662` CPC-group technologies.
- Evaluation uses `4` time periods and ranking metrics `NDCG@10`, `NDCG@20`, `NDCG@50`, and `NDCG@100`; the paper reports that DTF outperforms DTT-only, PC-DTT, CT-DTT, Tucker, CP, LR, and Patent Indicator in most settings.
- The Hughes Network Systems case study shows DTF correctly placing `H04L`, `H04W`, `H04B`, and `H03M` at the top of the predicted top-10 list, aligning with the leading technologies in the 2016 ground truth.
- In the same case study, the paper highlights that `H04Q` shares `39,898` common patents with `H04W`, illustrating how CTR can surface collaborative technologies beyond a firm's own recent portfolio.

## Limitations

- The paper reports most comparative performance through plots rather than a full numeric table, so exact NDCG deltas are hard to recover from text alone.
- Several crucial hyperparameters are left underspecified in the paper body, including concrete values for `m`, `n`, `d`, `d_0`, `d_1`, `d_2`, and `lambda`, which weakens reproducibility.
- The evaluation only covers firms with at least `200` patents and predicts at CPC-group granularity, so the method may not transfer to small firms or finer-grained technology codes.
- PCR and CTR rely on hand-crafted top-`m` / top-`n` neighbor selection and heuristic similarity functions; the framework does not compare against learned graph-relational alternatives.
- The evidence is entirely patent-centric, so the model ignores non-patent R&D signals such as hiring, products, acquisitions, or publications.

## Concepts Extracted

- [[technology-forecasting]]
- [[patent-mining]]
- [[technology-tracing]]
- [[potential-competitor-recognition]]
- [[collaborative-technology-recognition]]
- [[technology-distribution-forecasting]]
- [[convolutional-neural-network]]
- [[gated-recurrent-unit]]
- [[bayesian-personalized-ranking]]
- [[pairwise-ranking-loss]]

## Entities Extracted

- [[han-wu]]
- [[kun-zhang-ustc]]
- [[guangyi-lv]]
- [[qi-liu]]
- [[runlong-yu]]
- [[weihao-zhao]]
- [[enhong-chen]]
- [[jianhui-ma]]
- [[university-of-science-and-technology-of-china]]
- [[tensorflow]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
