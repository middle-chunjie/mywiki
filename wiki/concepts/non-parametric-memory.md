---
type: concept
title: Non-Parametric Memory
slug: non-parametric-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [external memory, retrieved memory]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Non-Parametric Memory** (非参数记忆) — external knowledge accessed at inference time through retrieval or lookup rather than being encoded directly in model weights.

## Key Points

- In the paper, retrieved Wikipedia passages are the primary non-parametric memory source.
- Non-parametric memory is especially helpful for questions about unpopular entities and other long-tail facts.
- Retrieval quality strongly affects usefulness: harmful cases have very low `recall@1`, showing that bad non-parametric memory can mislead the LM.
- The paper frames non-parametric memory as complementary to, not a total replacement for, parametric memory.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mallen-2023-when-2212-10511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mallen-2023-when-2212-10511]].
