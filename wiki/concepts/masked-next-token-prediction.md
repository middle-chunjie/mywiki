---
type: concept
title: Masked Next Token Prediction
slug: masked-next-token-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [MNTP, masked next token prediction]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Masked Next Token Prediction** — a training objective for decoder-style models that masks selected tokens but predicts each masked token from the representation at the previous position, preserving alignment with autoregressive pretraining.

## Key Points

- MNTP is the paper's main adaptation step for making bidirectional attention usable on decoder-only LLMs.
- The model predicts masked token `x_i` from position `i - 1`, not from the masked position itself.
- The paper searches masking strategies and finds `20%` BERT-style masking best for three backbones, while Mistral-7B prefers `80%` RoBERTa-style masking on sentence tasks.
- MNTP alone already delivers large gains on word-level tasks and strong improvements on MTEB.
- All MNTP runs use LoRA with `r = 16`, `alpha = 32`, and `1000` training steps.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
