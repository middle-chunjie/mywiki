---
type: concept
title: Decision Transformer
slug: decision-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [Decision Transformer]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Decision Transformer** — a Transformer-based policy model that casts reinforcement learning as sequence modeling over states, actions, and returns.

## Key Points

- The paper tests LASER on a `6`-layer decision Transformer trained for Sokoban policy learning.
- In this non-text setting, LASER improves accuracy from `50.67` to `53.0` and return from `0.575` to `0.965`.
- The gains are smaller than on language tasks, so the paper treats cross-domain generality as suggestive rather than settled.
- This experiment frames LASER as a broader intervention on Transformer computations rather than a purely language-specific trick.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sharma-2023-truth-2312-13558]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sharma-2023-truth-2312-13558]].
