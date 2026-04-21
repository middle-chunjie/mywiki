---
type: concept
title: Self-Information
slug: self-information
date: 2026-04-20
updated: 2026-04-20
aliases: [surprisal, information content, 自信息量]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Self-Information** (自信息量) — the quantity `-log P(x)` measuring how surprising or informative an event is, instantiated in language modeling as token-level negative log probability conditioned on prior context.

## Key Points

- [[li-2023-compressing]] uses token self-information `I(x_i) = -log_2 P(x_i | x_{<i})` as the core salience score for pruning context.
- Low-self-information spans are treated as more redundant because the target LLM is more likely to infer them from surrounding context or prior knowledge.
- The paper relies on additivity, summing token scores to obtain phrase- or sentence-level scores without training an extra scorer.
- In experiments, self-information is computed sentence by sentence to reduce a positional bias that otherwise depresses later-token scores.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-compressing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-compressing]].
