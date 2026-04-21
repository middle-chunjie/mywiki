---
type: concept
title: Retrieval Timing
slug: retrieval-timing
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval timing, 检索时机判断]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Timing** (检索时机判断) — the decision process that determines whether a model should trigger retrieval for a particular input and at what stage of generation.

## Key Points

- The paper treats retrieval timing as the central control problem in RAG.
- UAR decomposes retrieval timing into four orthogonal binary criteria instead of one scalar confidence heuristic.
- UAR-Criteria orders these decisions so that explicit user retrieval intent and temporal sensitivity can override simpler defaults.
- Better retrieval timing improves downstream accuracy while reducing unnecessary latency and irrelevant context injection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-unified-2406-12534]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-unified-2406-12534]].
