---
type: concept
title: Active Retrieval
slug: active-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [active retrieval, 主动检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Active Retrieval** (主动检索) — a retrieval strategy that decides whether and when to invoke external retrieval for each input instead of retrieving by default.

## Key Points

- UAR frames active retrieval as a retrieval-timing problem rather than a fixed always-retrieve policy.
- The paper argues active retrieval must consider user intent, knowledge intensity, temporal sensitivity, and model self-knowledge together.
- UAR unifies these cases into lightweight classification heads over frozen LLM representations.
- Empirically, active retrieval improves both answer quality and efficiency only when retrieval calls are selective and well timed.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-unified-2406-12534]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-unified-2406-12534]].
