---
type: concept
title: Memorization
slug: memorization
date: 2026-04-20
updated: 2026-04-20
aliases: [training-data memorization, 记忆化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Memorization** (记忆化) — the phenomenon where a model reproduces specific training data sequences rather than only generalizing abstract patterns from them.

## Key Points

- The paper adopts the Carlini et al. definition: a string is `(k, l)`-memorized if prompting with `k` tokens causes the model to generate the next `l` training tokens correctly.
- Pythia studies this with `k = 32` and `l = 32`, using the first `64` tokens from each training context.
- The observed distribution of memorized sequences across training batches is well fit by a Poisson point process.
- This result argues that simple placement of sensitive sequences early or late in training is not enough to reliably control memorization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[biderman-2023-pythia-2304-01373]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[biderman-2023-pythia-2304-01373]].
