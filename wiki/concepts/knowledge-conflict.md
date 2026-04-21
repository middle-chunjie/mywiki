---
type: concept
title: Knowledge Conflict
slug: knowledge-conflict
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge conflicts, 知识冲突]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Conflict** (知识冲突) — a situation where a model's internal knowledge and retrieved external evidence support incompatible answers to the same query.

## Key Points

- The paper identifies knowledge conflict as a central bottleneck in post-retrieval RAG rather than a rare corner case.
- Conflict frequency is tightly correlated with low retrieval precision under realistic web retrieval.
- In the authors' controlled analysis with Claude, `19.2%` of examples exhibit conflicts between no-RAG and RAG answers.
- Correct evidence is split across both sources in conflicting cases: internal knowledge is right `47.4%` of the time and external knowledge `52.6%`.
- Astute RAG addresses this by explicitly comparing consistent and conflicting evidence groups instead of trusting one source globally.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-astute-2410-07176]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-astute-2410-07176]].
