---
type: concept
title: Multi-token Prediction
slug: multi-token-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [MTP, 多词元预测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-token Prediction** (多词元预测) — a language-model training objective that predicts multiple future tokens in parallel from the same contextual representation instead of only the immediate next token.

## Key Points

- The paper implements MTP with a shared Transformer trunk plus `n` independent output heads, one for each future position `t + i`.
- The loss generalizes next-token training from `L_1` to `L_n`, forcing hidden states to encode information useful for several future steps at once.
- For fair comparisons, the authors keep total parameter count fixed by moving `n - 1` layers from the trunk into the auxiliary prediction heads.
- On code generation, MTP becomes more effective as model size increases; at `13B`, it yields sizable gains on MBPP and HumanEval over next-token baselines.
- The same extra heads can later be reused for self-speculative decoding, so the training objective also enables faster inference.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gloeckle-2024-better-2404-19737]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gloeckle-2024-better-2404-19737]].
