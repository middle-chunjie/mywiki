---
type: source
subtype: paper
title: Scaling Laws for Fine-Grained Mixture of Experts
slug: krajewski-2024-scaling-2402-07871
date: 2026-04-20
language: en
tags: [moe, scaling-laws, llm, transformers, efficiency]
processed: true

raw_file: raw/papers/krajewski-2024-scaling-2402-07871/paper.pdf
raw_md: raw/papers/krajewski-2024-scaling-2402-07871/paper.md
bibtex_file: raw/papers/krajewski-2024-scaling-2402-07871/paper.bib
possibly_outdated: false

authors:
  - Jakub Krajewski
  - Jan Ludziejewski
  - Kamil Adamczewski
  - Maciej Pióro
  - Michał Krutul
  - Szymon Antoniak
  - Kamil Ciebiera
  - Krystian Król
  - Tomasz Odrzygóźdź
  - Piotr Sankowski
  - Marek Cygan
  - Sebastian Jaszczur
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.07871
doi: 10.48550/arXiv.2402.07871
url: https://arxiv.org/abs/2402.07871
citation_key: krajewski2024scaling
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

The paper studies how sparse [[mixture-of-experts]] language models scale when expert size is no longer fixed to match the dense feed-forward layer. It introduces granularity `G`, which shrinks each expert while routing each token to `G` experts so the number of active parameters stays constant, and derives a joint scaling law over model size `N`, training tokens `D`, and `G`. Across more than `100` decoder-only Transformer runs on C4, the authors report that fine-grained MoE consistently beats dense Transformers when training duration is also optimized, reversing prior conclusions drawn under fixed-token comparisons. Their fitted law predicts increasing compute advantage for MoE at larger budgets and argues that the standard `G = 1` design is rarely compute-optimal.

## Problem & Motivation

Prior MoE scaling analyses often fixed training duration or fixed expert size, which can make sparse models look worse than dense Transformers at large scale simply because the MoE models are under-trained or constrained to suboptimal expert granularity. This paper asks whether MoE remains attractive under compute-optimal training and whether the usual choice of setting each expert to the same width as the dense FFN is actually optimal. The motivation is practical: if LLM training budgets continue to increase, sparse architectures are only compelling if their efficiency advantage survives when both model size and token count are tuned fairly.

## Method

- The authors study decoder-only [[decoder-only-transformer]] language models where each FFN block is replaced by a [[mixture-of-experts]] layer and compare them with dense Transformers trained on C4 using a GPT-2 tokenizer.
- They introduce [[granularity]] as `G = d_ff / d_expert`, where increasing `G` makes each expert narrower and routes each token to `G` experts so active parameters per token remain approximately constant.
- They define [[expansion-rate]] as `E = N_MoE / N_ff`, with the number of experts given by `N_expert = G * E`; the main sweep fixes `E = 64` and varies `G` from `1` to `16`.
- For fixed `N` and `D`, they fit granularity-dependent loss as `` `L_{N,D}(G) = g_{N,D} / G^{\gamma_{N,D}} + h_{N,D}` `` and combine this with a [[scaling-law]]-style Chinchilla form to obtain `` `L(N,D,G) = c + (g / G^\gamma + a) / N^\alpha + b / D^\beta` ``.
- The empirical study runs more than `100` MoE experiments with non-embedding model sizes from `129M` to `3.7B` parameters and token budgets from `16B` to `130B`; dense baselines are trained in parallel.
- Training uses AdamW with weight decay `0.1`, max learning rate `2e-4`, linear warmup for `1%` of steps, cosine decay to `2e-5`, batch size `0.5M` tokens, sequence length `2048`, and mixed precision with attention and router kept in high precision.
- The MoE implementation uses [[expert-choice-routing]] with token groups of size `256`, softmax over experts, an extra layer norm after the MoE output, and matched active FLOPs against dense models.
- To derive [[compute-optimal-scaling]], they model routing-aware compute as `` `F = (12 d_model^2 c_f + d_model E G c_r) * D * n_blocks` `` with `c_f = 6`, `c_r = 14`, and assume `d_model = 64 * n_blocks`, then solve for optimal `N`, `D`, and `G` under fixed `F` using Brent's method.
- The analysis explicitly models [[routing-overhead]]: extreme `G` can degrade wall-clock efficiency because router FLOPs, communication, and memory grow with the number of granular experts.

## Key Results

- The fitted MoE law achieves `RMSE = 0.015` and validation `RMSE = 0.019`, supporting the joint form over `N`, `D`, and `G`.
- Fitted MoE coefficients are `a = 18.1`, `alpha = 0.115`, `b = 30.8`, `beta = 0.147`, `g = 2.1`, `gamma = 0.58`, `c = 0.47`; dense coefficients are `a = 16.3`, `alpha = 0.126`, `b = 26.7`, `beta = 0.127`, `c = 0.47`.
- Under the fitted compute-optimal allocation, an MoE trained with `10^20` FLOPs is predicted to match a dense Transformer trained with roughly `20x` more compute, and the gap exceeds `40x` after `10^25` FLOPs.
- Optimal granularity rises with budget: `64 x 100M` active parameters prefers `G = 8` at `4.37B` tokens and `2.95e18` FLOPs, while `64 x 1B` prefers `G = 16` at `28.94B` tokens and `1.93e20` FLOPs.
- Larger budgets continue the trend: `64 x 7B` uses `G = 32`, `137.60B` tokens, and `6.46e21` FLOPs for loss `2.076`; `64 x 1T` uses `G = 64`, `7.94T` tokens, and `4.97e25` FLOPs for loss `1.367`.
- The paper argues that once `N`, `D`, and `G` are all optimized jointly, MoE remains more efficient than dense Transformers at every compute budget considered, overturning earlier fixed-duration conclusions.

## Limitations

- The experiments are limited to pretraining loss on C4 and do not test downstream tasks, instruction tuning, or inference quality, so the practical advantage is inferred rather than end-task verified.
- Most of the analysis fixes `E = 64`; the appendix includes a smaller `E = 16` study, but the paper does not fully map the joint interaction between expansion rate and granularity.
- The largest trained models are still far below frontier-scale systems, so claims about `10^23` to `10^25` FLOP regimes rely on extrapolating fitted scaling laws.
- Very high granularity can be counterproductive because routing and communication costs dominate, which means the theoretical gain is implementation-sensitive.

## Concepts Extracted

- [[mixture-of-experts]]
- [[scaling-law]]
- [[granularity]]
- [[expansion-rate]]
- [[expert-choice-routing]]
- [[compute-optimal-scaling]]
- [[routing-overhead]]
- [[decoder-only-transformer]]
- [[large-language-model]]

## Entities Extracted

- [[jakub-krajewski]]
- [[jan-ludziejewski]]
- [[kamil-adamczewski]]
- [[maciej-pioro]]
- [[michal-krutul]]
- [[szymon-antoniak-uw]]
- [[kamil-ciebiera]]
- [[krystian-krol]]
- [[tomasz-odrzygozdz]]
- [[piotr-sankowski]]
- [[marek-cygan]]
- [[sebastian-jaszczur]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
