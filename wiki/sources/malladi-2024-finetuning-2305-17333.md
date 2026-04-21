---
type: source
subtype: paper
title: Fine-Tuning Language Models with Just Forward Passes
slug: malladi-2024-finetuning-2305-17333
date: 2026-04-20
language: en
tags: [llm, fine-tuning, optimization, peft, memory-efficiency]
processed: true

raw_file: raw/papers/malladi-2024-finetuning-2305-17333/paper.pdf
raw_md: raw/papers/malladi-2024-finetuning-2305-17333/paper.md
bibtex_file: raw/papers/malladi-2024-finetuning-2305-17333/paper.bib
possibly_outdated: false

authors:
  - Sadhika Malladi
  - Tianyu Gao
  - Eshaan Nichani
  - Alex Damian
  - Jason D. Lee
  - Danqi Chen
  - Sanjeev Arora
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2305.17333
doi:
url: http://arxiv.org/abs/2305.17333
citation_key: malladi2024finetuning
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

The paper introduces MeZO, a memory-efficient adaptation of zeroth-order optimization for language-model fine-tuning that requires only forward passes and keeps memory usage near inference-time cost. Instead of storing perturbation vectors or activation graphs for backpropagation, MeZO replays the same random perturbation from a seed and estimates a projected gradient from two loss evaluations, letting a single 80GB A100 tune models as large as 30B parameters. Across RoBERTa-large and OPT models up to 66B, MeZO consistently beats zero-shot, in-context learning, and linear probing, while often approaching or matching full fine-tuning. The paper also argues theoretically that slowdown depends on a local effective-rank condition rather than raw parameter dimension, explaining why forward-only tuning can remain practical at large scale.

## Problem & Motivation

The paper targets the memory bottleneck of full-parameter language-model fine-tuning. Standard backpropagation must retain activations, gradients, and optimizer state, which in the authors' measurements can raise memory usage to roughly `12x` inference cost for large models. This makes adaptation of 10B+ parameter models inaccessible on modest hardware even when plain inference is feasible.

The authors revisit zeroth-order optimization because it estimates update directions from loss values alone and therefore avoids backward passes. Classical analyses suggest such methods should become prohibitively slow in high dimensions, so the core question is whether a carefully engineered forward-only optimizer can still fine-tune large pretrained language models efficiently in practice.

## Method

- **SPSA estimator**: MeZO starts from simultaneous perturbation stochastic approximation, using `\hat{\nabla}\mathcal{L}(\theta; \mathcal{B}) = ((\mathcal{L}(\theta + \epsilon z; \mathcal{B}) - \mathcal{L}(\theta - \epsilon z; \mathcal{B})) / (2\epsilon)) z` with `z ~ \mathcal{N}(0, I_d)` and default `n = 1`.
- **In-place MeZO update**: rather than storing `z \in \mathbb{R}^d`, the algorithm samples a random seed `s`, regenerates the same perturbation four times, computes `\ell_+` and `\ell_-` from two forward passes, resets parameters, and applies `\theta_{t+1} = \theta_t - \eta_t \cdot ((\ell_+ - \ell_-) / (2\epsilon)) \cdot z`.
- **Memory profile**: this seed-replay design reduces ZO-SGD memory from roughly `2x` inference to essentially inference-equivalent memory; when perturbing weight matrices instead of individual scalars, extra memory is bounded by the largest matrix, reported as `0.86 GB` for OPT-66B embeddings.
- **Checkpoint/storage efficiency**: MeZO can reconstruct the optimization trajectory from a seed plus saved projected gradients. For a `66B` model and `20,000` steps, the paper reports less than `0.1 MB` checkpoint storage, versus `38 MB` for LoRA and `12 MB` for prefix tuning.
- **Training setup**: RoBERTa-large experiments use `100K` MeZO steps versus `1K` FT steps; OPT experiments use `20K` MeZO steps versus `5` FT epochs (`625` steps in the 13B setting). MeZO uses constant learning rates, whereas FT uses linear schedules.
- **Hyperparameters**: for OPT full-parameter MeZO, batch size is `16`, learning rates are `{1e-6, 1e-7}` (or `{1e-6, 5e-7, 1e-7}` on SQuAD/DROP), and perturbation scale is `\epsilon = 1e-3`; for prefix tuning, batch size is `16`, `\epsilon = 1e-1`, learning rates are `{1e-2, 1e-3}` or `{5e-2, 1e-2, 5e-3}` on QA.
- **PEFT compatibility**: MeZO is applied to both full-parameter tuning and [[parameter-efficient-fine-tuning]] variants. For [[lora]], the paper uses rank/scale `(r, \alpha) = (8, 16)`; for [[prefix-tuning]], it uses `m = 5` prefix tokens and initializes prefixes from real token activations instead of random initialization.
- **Task-specific training/inference**: classification uses label-word or candidate log-likelihoods; multiple-choice and QA retain loss only on candidate/answer tokens; QA training uses [[teacher-forcing]] and inference uses greedy decoding. The ICL baseline uses `32` in-context examples.
- **Theory**: the analysis assumes favorable post-pretraining geometry and shows per-step descent and global convergence slowdowns scaling with local [[effective-rank]] `r` rather than parameter count `d`, including a PL-style global result.

## Key Results

- On OPT-13B with `1,000` training examples, MeZO is within `1%` of or better than full fine-tuning on `7/11` tasks while using about `1/12` the memory; for example, MeZO reaches `91.4` on SST-2, `88.0` on COPA, and `84.7` on SQuAD.
- On larger models, MeZO continues to scale: OPT-30B reaches `90.6` on SST-2 and `85.2` on SQuAD, while OPT-66B with MeZO/prefix reaches `93.6` on SST-2 and `85.0` on SQuAD, comfortably above zero-shot and ICL baselines.
- The hardware advantage is large: with `1x A100 80GB`, MeZO can tune `30B` models, whereas full FT fits only `2.7B`; with `2x A100`, MeZO reaches `66B` while full FT fits `6.7B`.
- Memory profiling on OPT shows MeZO uses the same memory as zero-shot inference and saves up to `12x` versus full FT and `6x` versus prefix FT.
- Wall-clock analysis on OPT-30B reports `5.896 s/step` for MeZO (`1` GPU) versus `45.608 s/step` for FT (`8` GPUs), a `7.74x` per-step speedup; despite needing `32x` more steps, MeZO still uses roughly half the GPU-hours overall.
- MeZO can optimize non-differentiable objectives: on Table 3, optimizing accuracy/F1 with forward-only updates still beats zero-shot, e.g. `92.7` on SST-2 and `78.5` on SQuAD for the reported settings.

## Limitations

- MeZO is not a drop-in speed winner in iteration count: it often needs far more optimization steps than backpropagation (`20K` vs `625` on OPT, `100K` vs `1K` on RoBERTa).
- Performance gaps remain on harder tasks; for OPT-13B, MeZO trails FT substantially on CB (`67.9` vs `83.9`) and MultiRC (`60.1` vs `71.1`), so memory savings do not uniformly preserve task quality.
- The method is prompt-dependent; the paper explicitly states that MeZO only worked reliably when prompt-based formulations were used.
- The theoretical story depends on favorable geometry assumptions such as low local effective rank and PL-like behavior after pretraining, which may not hold for all objectives or adaptation regimes.
- The study does not evaluate combinations with other memory-saving training methods such as gradient checkpointing, FlashAttention, or quantization, leaving open whether the best practical stack is MeZO alone or MeZO plus complementary systems tricks.

## Concepts Extracted

- [[zeroth-order-optimization]]
- [[simultaneous-perturbation-stochastic-approximation]]
- [[gradient-free-optimization]]
- [[parameter-efficient-fine-tuning]]
- [[lora]]
- [[prefix-tuning]]
- [[in-context-learning]]
- [[linear-probing]]
- [[effective-rank]]
- [[teacher-forcing]]
- [[autoregressive-language-model]]

## Entities Extracted

- [[sadhika-malladi]]
- [[tianyu-gao]]
- [[eshaan-nichani]]
- [[alex-damian]]
- [[jason-d-lee]]
- [[danqi-chen]]
- [[sanjeev-arora]]
- [[princeton-university]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
