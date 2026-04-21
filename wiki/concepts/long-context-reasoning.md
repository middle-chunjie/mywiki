---
type: concept
title: Long-Context Reasoning
slug: long-context-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [long context reasoning, long-range context reasoning, 长上下文推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Context Reasoning** (长上下文推理) — reasoning that requires processing long inputs and integrating distant, sparse, or distracting evidence rather than relying on short local cues.

## Key Points

- BBEH increases the macro average context length to about `6x` that of BBH.
- Tasks such as Object Counting, Shuffled Objects, Buggy Tables, and Temporal Sequences explicitly stress long-context memory and multi-needle retrieval.
- The benchmark is designed so distractors and irrelevant updates make naive local heuristics unreliable.
- The paper finds that gains from reasoning-specialized models tend to increase as task context length grows.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kazemi-2025-bigbench-2502-19187]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kazemi-2025-bigbench-2502-19187]].
