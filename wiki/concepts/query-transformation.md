---
type: concept
title: Query Transformation
slug: query-transformation
date: 2026-04-20
updated: 2026-04-20
aliases: [query transformation, query reformulation, 查询变换]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query Transformation** (查询变换) — the process of rewriting or decomposing an input question into alternative search queries that improve evidence retrieval for downstream reasoning.

## Key Points

- AirRAG treats query transformation as a first-class reasoning action `A_4` rather than an implicit side effect of chain-of-thought prompting.
- The action supports multiple forms of transformation, including rewriting, step-back prompting, follow-up sub-questions, and multi-query retrieval.
- In the tree-search framework, QT can be interleaved with RA to expand the search space beyond a single chain of queries.
- The paper finds QT is one of the diversity-sensitive actions where increasing output multiplicity and using `top-p = 1.0`, `temperature = 1.0` is especially helpful.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[feng-2025-airrag-2501-10053]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[feng-2025-airrag-2501-10053]].
