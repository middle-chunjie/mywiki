---
type: concept
title: False Pruning
slug: false-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [premature pruning error, 错误剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**False Pruning** (错误剪枝) — a constrained decoding failure mode in which correct generation paths are discarded early because locally plausible prefixes later prove grounded in irrelevant documents.

## Key Points

- RetroLLM identifies false pruning as the main reason naive corpus-level constrained evidence generation performs poorly.
- The problem is especially severe in early decoding steps, where a large corpus creates too many misleading prefix choices.
- The paper's empirical study shows prefix relevance drops sharply under corpus-level constraints but degrades much less when decoding is restricted to relevant documents.
- Hierarchical FM-index constraints and forward-looking decoding are introduced specifically to mitigate this failure mode.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-retrollm-2412-11919]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-retrollm-2412-11919]].
