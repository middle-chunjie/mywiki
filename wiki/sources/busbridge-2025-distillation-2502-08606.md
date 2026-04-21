---
type: source
subtype: paper
title: Distillation Scaling Laws
slug: busbridge-2025-distillation-2502-08606
date: 2026-04-20
language: en
tags: [distillation, scaling-laws, language-models, pretraining, compute]
processed: true

raw_file: raw/papers/busbridge-2025-distillation-2502-08606/paper.pdf
raw_md: raw/papers/busbridge-2025-distillation-2502-08606/paper.md
bibtex_file: raw/papers/busbridge-2025-distillation-2502-08606/paper.bib
possibly_outdated: false

authors:
  - Dan Busbridge
  - Amitis Shidani
  - Floris Weers
  - Jason Ramapuram
  - Etai Littwin
  - Russ Webb
year: 2025
venue: ICML
venue_type: conference
arxiv_id: 2502.08606
doi: 10.48550/arXiv.2502.08606
url: http://arxiv.org/abs/2502.08606
citation_key: busbridge2025distillation
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper develops a distillation scaling law for language-model pretraining that predicts student validation cross-entropy from student size `N_S`, distillation tokens `D_S`, and teacher quality summarized by teacher cross-entropy `L_T`. Using controlled transformer experiments spanning `143M` to `12.6B` parameters on English [[c4]], the authors show that teacher size and teacher data matter to the student only through `L_T`, and that the teacher-student interaction follows a broken-power-law regime with a [[capacity-gap]]: beyond an optimal teacher quality, a stronger teacher can hurt the student. The fitted law then supports compute-allocation analysis across best-case, teacher-inference, and teacher-training scenarios. Main takeaway: distillation is attractive when a teacher already exists or will be amortized across many students, but supervised pretraining remains asymptotically preferable when end-to-end compute must also pay for training the teacher.

## Problem & Motivation

Compute-optimal language models become larger as available training FLOPs increase, which improves loss but makes inference increasingly expensive. An alternative is [[overtraining]] small models on many more tokens, but that is itself costly and slow. The paper asks whether [[knowledge-distillation]] can deliver the capability of small, heavily trained models at lower training cost, and how compute should be allocated between teacher training, teacher inference, and student training. It also resolves contradictory prior claims by making teacher strength, student size, and token budgets explicit in one scaling-law framework.

## Method

- The paper starts from a supervised [[model-scaling]] law, `L(N, D) = E + (A / N^alpha + B / D^beta)^gamma`, and uses it to summarize teacher quality by `L_T = L(N_T, D_T)` rather than treating teacher size and teacher tokens as separate drivers.
- Distillation uses autoregressive next-token modeling: student training minimizes `L_S = (1 - lambda) L_NTP + lambda L_KD + lambda_Z L_Z`, but the main experiments use pure distillation with `lambda = 1`, temperature `tau = 1`, and `Z-loss = 1e-4`.
- The proposed distillation scaling law is a broken-power-law function of teacher quality: `L_S(N_S, D_S, L_T) = L_T + L_T^(-c0) (1 + (L_T / (d1 * L~_S))^(1 / f1))^(-c1 f1) (A' / N_S^alpha' + B' / D_S^beta')^gamma'`, where `L~_S = L(N_S, D_S)` is the supervised student loss.
- Three experiment families identify the coefficients: fixed-`M` teachers with student IsoFLOPs, teacher IsoFLOPs with fixed-`M` students, and fixed-`M` teacher/student sweeps that expose the [[capacity-gap]].
- Models are transformer-based [[decoder-only-language-model]]s trained with `n_ctx = 4096`, pre-norm RMSNorm, multi-head attention, [[rotary-positional-embedding]] with base frequency `500k`, fixed aspect ratio `d_model / n_layers = 128`, FFN ratio `rho_ffn = 8 / 3`, and gated linear activations.
- The model family spans `143M` to `12.6B` non-embedding parameters; the main teacher profiles include fixed-`M ~= 20` teachers from `198M` to `7.75B`, while students and teachers are evaluated under multiple IsoFLOP budgets such as `3e19`, `1e20`, `3e20`, and `1e21` FLOPs.
- Training is implemented in [[axlearn]] with simplified `muP`, decoupled weight decay `1e-4`, fixed peak learning rate `1e-2` across scales, cosine decay, `2000` warmup steps, final learning rate `1e-3` of peak, and gradient clipping to global norm `1.0`.
- Data comes from the English-only split of [[c4]]; the extracted appendix states the corpus has about `180B` tokens total, split into `90B` unique teacher tokens and `90B` unique student tokens, with data repetition only for sufficiently overtrained large-token runs.
- For compute accounting, distillation cost is approximated as `3 F(N_S) D_S + F(N_T) (delta_Lgt D_S + 3 delta_Pre D_T)`, and the paper studies four cases: fully amortized teacher, teacher inference only, teacher pretraining only, and teacher pretraining plus inference.

## Key Results

- The supervised and distillation scaling laws fit their observations with less than `1%` relative prediction error, including extrapolation from weaker to stronger models.
- The experiments show that student loss depends on teacher size and teacher data only through teacher cross-entropy `L_T`, which collapses multiple IsoFLOP teacher profiles onto a smooth curve.
- The [[capacity-gap]] is empirically visible: for fixed-`M` students with `M_S in {20, 40, 80, 160, 320}`, student cross-entropy first decreases and then increases as teacher size grows, so a better teacher can eventually make the student worse.
- Best-case distillation can beat supervised learning at modest token budgets, but the advantage disappears when student tokens become large; the paper visualizes this over student sizes `546M` to `7.75B` and token budgets from `1B` to `10T`.
- In teacher-selection analyses with `D_S in {250B, 1T, 4T, 16T}`, the optimal teacher cross-entropy decreases roughly as a power law with student size rather than being determined by raw teacher parameter count.
- In compute-optimal analyses for students of `300M`, `1B`, `3B`, and `10B`, supervised learning eventually matches or beats best-case distillation at sufficiently large compute, while teacher-pretraining-inclusive scenarios remain worse than supervised training when producing only one student.
- The paper also quantifies FLOP estimation error: the common `2N` forward-FLOPs approximation underestimates true cost by about `7.42%` to `39.74%` across their model family, while the corrected `2N(1 + c1 N^(-1/3) + c2 N^(-2/3))` approximation reduces error to at most about `0.39%`.

## Limitations

- The study is limited to language modeling on English [[c4]], so the fitted coefficients are not claimed to transfer unchanged to other datasets, modalities, or domains.
- Large-token runs repeat data, and the paper argues from supervised evidence that moderate repetition is probably acceptable, but this is not directly proven for distillation.
- The experiments focus on full-distribution KL-style [[knowledge-distillation]], not sequence-level or hard distillation via sampled synthetic sequences, so the law may not cover those workflows.
- Teacher and student are trained on the same data distribution with transformer-family models; cross-domain distillation and architecture mismatch are explicitly left unresolved.
- Calibration analysis is diagnostic rather than definitive: the paper studies [[model-calibration]] to understand why stronger teachers can be harder to imitate, but does not derive the scaling law from first principles.

## Concepts Extracted

- [[knowledge-distillation]]
- [[model-scaling]]
- [[compute-optimal-training]]
- [[capacity-gap]]
- [[overtraining]]
- [[large-language-model]]
- [[small-language-model]]
- [[autoregressive-language-model]]
- [[decoder-only-language-model]]
- [[rotary-positional-embedding]]
- [[model-calibration]]

## Entities Extracted

- [[dan-busbridge]]
- [[amitis-shidani]]
- [[floris-weers]]
- [[jason-ramapuram]]
- [[etai-littwin]]
- [[russ-webb]]
- [[apple]]
- [[axlearn]]
- [[c4]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
