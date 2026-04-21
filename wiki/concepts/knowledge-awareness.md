---
type: concept
title: Knowledge Awareness
slug: knowledge-awareness
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge-aware, knowledge awareness, 知识感知]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Awareness** (知识感知) — the ability to distinguish instructions that require factual external knowledge from those that mainly rely on reasoning, creativity, or input-local information.

## Key Points

- UAR uses knowledge awareness to avoid retrieval on non-knowledge-intensive tasks such as math reasoning or reading comprehension with provided context.
- The paper constructs knowledge-aware data from Self-RAG non-retrieval instructions plus knowledge-intensive examples from time-aware datasets.
- This criterion helps explain why always-retrieve systems underperform on DROP and GSM8K.
- UAR's knowledge-aware accuracy exceeds `90` on both 7B and 13B AR-Bench evaluations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-unified-2406-12534]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-unified-2406-12534]].
