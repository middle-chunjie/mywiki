---
type: source
subtype: paper
title: "TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning"
slug: liu-2023-timesurl
date: 2026-04-20
language: en
tags: [time-series, self-supervised-learning, contrastive-learning, representation-learning, forecasting]
processed: true

raw_file: raw/papers/liu-2023-timesurl/paper.pdf
raw_md: raw/papers/liu-2023-timesurl/paper.md
bibtex_file: raw/papers/liu-2023-timesurl/paper.bib
possibly_outdated: false

authors:
  - Jiexi Liu
  - Songcan Chen
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2312.15709
doi:
url: http://arxiv.org/abs/2312.15709
citation_key: liu2023timesurl
paper_type: method

read_status: unread

domain: time-series
---

## Summary

TimesURL proposes a universal time-series representation learning framework that adapts self-supervised contrastive learning to temporal data rather than directly borrowing image or NLP recipes. The method combines frequency-temporal augmentation (FTAug), double Universum hard negatives, and a masked time-reconstruction objective so the encoder captures both fine-grained segment information and coarse instance-level semantics. Concretely, it pairs random cropping with frequency mixing to preserve temporal structure, synthesizes temporal-wise and instance-wise hard negatives in embedding space, and jointly optimizes contrastive and reconstruction losses with a TCN backbone. Across six downstream task families, the paper reports consistent gains over prior self-supervised baselines such as TS2Vec and InfoTS, arguing that suitable augmentations, harder negatives, and multi-level objectives are all necessary for universal representations.

## Problem & Motivation

The paper targets universal time-series representation learning: one pretrained encoder should support forecasting, imputation, classification, anomaly detection, and transfer learning without task-specific pretraining. The authors argue that existing time-series contrastive methods fail for three reasons. First, augmentations borrowed from CV/NLP, such as flipping or permutation, can destroy trend and temporal dependency structure. Second, most negatives in time series are easy negatives because local smoothness and Markov structure make many mismatched segments obviously different, so contrastive training receives weak discrimination signals. Third, optimizing only segment-level or only instance-level information is insufficient because different downstream tasks require both timestamp-local and whole-series semantics.

## Method

- **Problem setup**: each input series is `x_i ∈ R^{T×F}` and the encoder learns representations `r_i = {r_{i,1}, ..., r_{i,T}}` with `r_{i,t} ∈ R^K`, then reuses them across downstream tasks.
- **FTAug positive-pair construction**: augmentation combines frequency mixing with random cropping. Crops satisfy `0 < a_1 ≤ a_2 ≤ b_1 ≤ b_2 ≤ T`, and optimization focuses on the overlapping interval `[a_2, b_1]` so matched timestamps remain context-consistent.
- **Frequency mixing**: selected FFT components from `x_i` are replaced by the same frequencies from another sample `x_k`, then mapped back with inverse FFT. The goal is to vary context without breaking temporal semantics as aggressively as sign flipping or permutation.
- **Temporal-wise Universums**: hard negatives are synthesized in embedding space as `r^temp_{i,t} = λ_1 r_{i,t} + (1 - λ_1) r_{i,t'}` and `r'^temp_{i,t} = λ_1 r'_{i,t} + (1 - λ_1) r'_{i,t'}`, where `t' ∈ Ω`, `t' ≠ t`, and `λ_1 ∈ (0, 0.5]`.
- **Instance-wise Universums**: a second hard-negative family mixes the anchor with another instance at the same timestamp: `r^inst_{i,t} = λ_2 r_{i,t} + (1 - λ_2) r_{j,t}` and `r'^inst_{i,t} = λ_2 r'_{i,t} + (1 - λ_2) r'_{j,t}`, with `j ≠ i` and `λ_2 ∈ (0, 0.5]`.
- **Dual contrastive objective**: temporal and instance losses `ℓ_temp^(i,t)` and `ℓ_inst^(i,t)` use positives `(r_{i,t}, r'_{i,t})` and negatives from both original samples and Universums. The combined loss is `L_dual = (1 / (|B|T)) Σ_i Σ_t (ℓ_temp^(i,t) + ℓ_inst^(i,t))`.
- **Hierarchical pooling**: TimesURL follows hierarchical contrastive learning with max pooling along the time axis so training captures multi-scale structure instead of only one granularity.
- **Masked time reconstruction**: masked inputs `x_M` and `x'_M` are encoded and reconstructed with MSE only on masked timestamps, `L_recon = (1 / (2|B|)) Σ_i ||m_i ⊙ (x̃_i - x_i)||_2^2 + ||m'_i ⊙ (x̃'_i - x'_i)||_2^2`.
- **Joint training and implementation details**: the final objective is `L = L_dual + α L_recon`, where `α` balances contrastive and reconstruction terms. The backbone encoder is a `Temporal Convolution Network (TCN)`, classification representations use dimension `320`, imputation masks `12.5%`, `25%`, `37.5%`, and `50%` of points, and forecasting is evaluated on horizons from `24` to `720`.

## Key Results

- **Classification**: on `30` UEA datasets, TimesURL reaches average accuracy `0.752` and average rank `1.367`, beating InfoTS at `0.714` by `+3.8%`; on `128` UCR datasets it reaches `0.845` vs. InfoTS `0.838`.
- **Imputation**: average performance across ETTh1, ETTh2, and ETTm1 is `MSE 1.326` and `MAE 0.860`, improving over InfoTS (`1.386`, `0.864`) and TS2Vec (`1.418`, `0.871`).
- **Forecasting**: the aggregate univariate forecasting average is `MSE 0.0977` and `MAE 0.2292`, better than CoST (`0.1021`, `0.2328`) and TS2Vec (`0.1156`, `0.2528`).
- **Anomaly detection**: Yahoo F1 improves slightly over TS2Vec from `0.745` to `0.749`; KPI F1 improves from `0.677` to `0.688`.
- **Transfer learning**: average downstream accuracy is `0.864` when transferring from CBF and `0.895` from CinCECGTorso, compared with `0.912` in the no-transfer setting, indicating moderate but incomplete transferability.
- **Ablation**: removing frequency mixing drops average UEA accuracy from `0.752` to `0.709` (`-4.3%`), removing double Universums drops it to `0.716` (`-3.6%`), and removing time reconstruction drops it to `0.735` (`-1.8%`).

## Limitations

- The main paper defers many implementation details to the appendix; crucial training hyperparameters such as the exact `α`, masking configuration beyond downstream evaluation ratios, and optimization settings are not fully specified in the provided markdown.
- The best-results claim is aggregate rather than universal: some individual forecasting cells are still matched or slightly beaten by CoST or other baselines.
- Evaluation focuses on standard academic benchmarks such as UEA, UCR, ETT, Yahoo, and KPI; robustness on noisier industrial, medical, or irregularly sampled multivariate settings is not deeply explored.
- The method is a 2023 arXiv preprint rather than a venue paper in the provided metadata, so the evidence here should be treated as promising but not final.

## Concepts Extracted

- [[representation-learning]]
- [[self-supervised-learning]]
- [[contrastive-learning]]
- [[hard-negative-sampling]]
- [[universal-representation-learning]]
- [[time-series-augmentation]]
- [[hierarchical-contrastive-loss]]
- [[time-reconstruction]]
- [[masked-autoencoding]]
- [[transfer-learning]]

## Entities Extracted

- [[jiexi-liu]]
- [[songcan-chen]]
- [[timesurl]]
- [[ts2vec]]
- [[infots]]
- [[temporal-convolutional-network]]
- [[uea-time-series-classification-archive]]
- [[ucr-time-series-archive]]
- [[ett-dataset]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
