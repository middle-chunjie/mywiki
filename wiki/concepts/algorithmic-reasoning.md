---
type: concept
title: Algorithmic Reasoning
slug: algorithmic-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [算法推理]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Algorithmic Reasoning** (算法推理) — the ability of a model to execute compositional symbolic procedures, such as arithmetic or program-like transformations, rather than only pattern-match local continuations.

## Key Points

- The paper evaluates algorithmic reasoning on polynomial arithmetic over `F_7[X] / (X^5)` with unary negation, addition, multiplication, and composition.
- Models are tested on both in-domain (`m <= 5` operations) and out-of-domain (`m > 5`) generalization, using greedy decoding on fixed `2000`-example test sets.
- Multi-token prediction improves performance across task difficulties, especially for out-of-distribution generalization.
- Replacing next-token training with MTP helps more than tripling model size from `30M` to `100M` in the authors' synthetic arithmetic setup.
- The advantage remains when pause tokens are inserted, suggesting the gain is not limited to a single prompt format.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gloeckle-2024-better-2404-19737]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gloeckle-2024-better-2404-19737]].
