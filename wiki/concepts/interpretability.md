---
type: concept
title: Interpretability
slug: interpretability
date: 2026-04-20
updated: 2026-04-20
aliases: [interpretability, 可解释性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Interpretability** (可解释性) — the ability to inspect a model or control mechanism and relate its internal parameters to understandable patterns in behavior or data.

## Key Points

- The paper analyzes the learned switch matrix `W` with SVD to identify dimensions most associated with toxicity-related word choices.
- It projects top singular directions back into token space and uses Perspective API to decide which sign of each direction corresponds to more toxic language.
- The resulting token lists make the conditioning effect more legible by surfacing words LM-Switch is suppressing or amplifying.
- This analysis supports the paper's claim that steering in embedding space can be more interpretable than opaque classifier-guided decoding methods.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
