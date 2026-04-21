---
type: concept
title: Instruction Negative
slug: instruction-negative
date: 2026-04-20
updated: 2026-04-20
aliases: [instruction negatives, instruction-negative mining]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Instruction Negative** — a retrieval training example where a passage remains relevant to the base query but becomes non-relevant once an added instruction changes the relevance criteria.

## Key Points

- Promptriever introduces instruction negatives to stop the retriever from ignoring the instruction while still matching the original query semantics.
- The paper generates `3` instruction-negative candidates per `(query, instruction)` pair with `gpt-4o-2024-05-13` and filters them using `FollowIR-7B`.
- The construction is intentionally instance-level rather than dataset-level: the same query can have different negative passages depending on the accompanying instruction.
- In ablations, adding instruction negatives on top of instruction-only training raises average `p-MRR` from `+5.7` to `+8.8`, showing they materially increase instruction sensitivity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weller-2024-promptriever-2409-11136]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weller-2024-promptriever-2409-11136]].
