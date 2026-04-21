---
type: concept
title: Knowledge Synthesis
slug: knowledge-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [evidence synthesis, knowledge construction, 知识合成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Synthesis** (知识合成) — the process of constructing a higher-quality evidence representation from noisy retrieved content before downstream generation.

## Key Points

- In BIDER, knowledge synthesis is the first training stage and produces oracle KSE from retrieved documents.
- The synthesis pipeline first retrieves candidate sentence nuggets with semantic similarity, then removes redundancy with iterative selection.
- A final cleaning step uses generator likelihood change to discard nuggets that are unhelpful or harmful to answer generation.
- Ablations show replacing the synthesis target with naive sentence retrieval causes a large performance drop, especially on `NQ`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-bider-2402-12174]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-bider-2402-12174]].
