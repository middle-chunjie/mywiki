---
type: source
subtype: paper
title: "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"
slug: biderman-2023-pythia-2304-01373
date: 2026-04-20
language: en
tags: [llm, scaling, pretraining, memorization, bias]
processed: true

raw_file: raw/papers/biderman-2023-pythia-2304-01373/paper.pdf
raw_md: raw/papers/biderman-2023-pythia-2304-01373/paper.md
bibtex_file: raw/papers/biderman-2023-pythia-2304-01373/paper.bib
possibly_outdated: true

authors:
  - Stella Biderman
  - Hailey Schoelkopf
  - Quentin Anthony
  - Herbie Bradley
  - "Kyle O'Brien"
  - Eric Hallahan
  - Mohammad Aflah Khan
  - Shivanshu Purohit
  - USVSN Sai Prashanth
  - Edward Raff
  - Aviya Skowron
  - Lintang Sutawika
  - Oskar van der Wal
year: 2023
venue: ICML 2023
venue_type: conference
arxiv_id: 2304.01373
doi: 10.48550/arXiv.2304.01373
url: http://arxiv.org/abs/2304.01373
citation_key: biderman2023pythia
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

Pythia introduces a controlled suite of decoder-only large language models for studying how behavior changes across both training time and scale. The paper releases 16 models spanning `70M` to `12B` parameters, with 8 sizes trained on the original Pile and 8 on a near-deduplicated variant, while preserving identical architecture choices and data ordering within each suite. Each model exposes `154` checkpoints, enabling fine-grained analysis of capability emergence, memorization, and bias. Using this setup, the authors report three main findings: targeted late-stage corpus interventions can reduce gender bias with limited utility loss, memorized sequences are well modeled by a Poisson point process rather than by position in training order, and term-frequency effects on downstream performance emerge mainly in larger models after substantial training progress.

## Problem & Motivation

Prior public LLM releases were poorly suited for scientific analysis of training dynamics because they typically lacked one or more of the following: public access, consistent training order across scales, dense intermediate checkpoints, or public data provenance. That made it difficult to test questions about when capabilities emerge, how memorization depends on exposure, and whether corpus statistics causally shape downstream behavior. Pythia is designed as a controlled model suite that removes confounds from architecture, data ordering, and checkpoint sparsity so researchers can study training-time phenomena directly rather than inferring them from a few final checkpoints.

## Method

- **Suite design**: `16` decoder-only autoregressive transformers, arranged as `8` sizes on the original Pile and `8` matched sizes on a deduplicated Pile; parameter scales span `70M, 160M, 410M, 1.0B, 1.4B, 2.8B, 6.9B, 12B`.
- **Scientific control**: all models within a suite use the same data ordering, checkpoint schedule, and broadly matched architecture so that training-time comparisons are not confounded by different corpora or optimization setups.
- **Training data**: the standard suite is trained for `≈ 300B` tokens; the deduplicated suite applies near-deduplication with MinHashLSH threshold `0.87`, producing a corpus of `≈ 207B` tokens and thus `≈ 1.5` epochs over the deduplicated data.
- **Architecture**: models are decoder-only Transformers trained with fully dense attention, untied embedding/unembedding matrices, rotary position embeddings with `rotary-pct = 0.25`, and parallel attention/MLP blocks following GPT-NeoX-style design choices.
- **Optimization**: Adam with `β1 = 0.9`, `β2 = 0.95`, `ε = 1e-8`, cosine decay, warmup fraction `0.01`, weight decay `0.01`, gradient clipping `1.0`, and global batch size `1024` sequences of length `2048`, i.e. `2,097,152` tokens per step.
- **Systems setup**: training uses GPT-NeoX with DeepSpeed ZeRO stage `1`, data and tensor parallelism, Flash Attention, FP16, and A100 40GB GPUs; larger models use model parallel sizes up to `4`.
- **Checkpointing**: save at initialization, at early log-spaced steps `{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}`, and then every `1000` iterations / `2,097,152,000` tokens, yielding `154` checkpoints per model.
- **Case studies**: the paper evaluates gender-bias interventions by swapping gendered pronouns in the last `7%` or `21%` of training, memorization using the Carlini et al. criterion with `(k, l) = (32, 32)`, and few-shot / QA frequency effects across checkpoints and scales.

## Key Results

- The released suite contains `16` models and `154` checkpoints per model, for `2,464` public checkpoints spanning `70M` to `12B` parameters.
- Full training for the paper required `544,280` A100-hours after retraining both the standard and deduplicated suites.
- Final zero-shot performance for the `12B` standard Pythia model reaches `0.705` on LAMBADA, `0.760` on PIQA, `0.702` on ARC-Easy, and `0.902` on SciQ.
- Final five-shot performance for the `12B` deduplicated Pythia model reaches `0.650` on LAMBADA, `0.757` on PIQA, `0.702` on ARC-Easy, `0.347` on ARC-Challenge, and `0.952` on SciQ.
- The authors identify a phase change after about `65,000` training steps (`≈ 45%` of training): models at `2.8B+` begin to show a clear correlation between task accuracy and pretraining term frequency, whereas smaller models largely do not.
- In the bias intervention study, resuming training from checkpoints `21B` tokens (`7%`) or `63B` tokens (`21%`) before the end reduces measured gender bias; at `6.9B`, the WinoBias intervention flips behavior from pro-stereotypical to anti-stereotypical.
- For memorization analysis, using the Carlini-style criterion with `k = 32` and `l = 32`, the distribution of memorized sequences over training batches is well fit by a Poisson point process, arguing against a simple training-order effect.

## Limitations

- The suite is intentionally controlled rather than SOTA-optimized, so its conclusions may not transfer cleanly to newer architectures, newer data mixtures, or post-2023 training recipes.
- Most evaluations are English-centric and benchmark-centric; the paper does not provide similarly deep analysis for multilingual, multimodal, or instruction-tuned settings.
- The deduplicated suite runs for `≈ 1.5` epochs over `207B` tokens, which complicates clean interpretation of deduplication effects near and after the second-epoch boundary.
- Several headline case-study claims are qualitative in the main text; exact effect sizes for bias and memorization are often shown in figures rather than summarized as compact tables.
- Because the paper centers on decoder-only pretraining, it does not address downstream alignment, RLHF, or instruction-tuning dynamics that now materially shape deployed LLM behavior.

## Concepts Extracted

- [[large-language-model]]
- [[training-dynamics]]
- [[model-scaling]]
- [[data-deduplication]]
- [[memorization]]
- [[few-shot-learning]]
- [[bias-mitigation]]
- [[rotary-positional-embedding]]
- [[flash-attention]]

## Entities Extracted

- [[stella-biderman]]
- [[hailey-schoelkopf]]
- [[quentin-anthony]]
- [[herbie-bradley]]
- [[kyle-obrien]]
- [[eric-hallahan]]
- [[mohammad-aflah-khan]]
- [[shivanshu-purohit]]
- [[usvsn-sai-prashanth]]
- [[edward-raff]]
- [[aviya-skowron]]
- [[lintang-sutawika]]
- [[oskar-van-der-wal]]
- [[eleutherai]]
- [[gpt-neox]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
