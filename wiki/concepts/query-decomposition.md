---
type: concept
title: Query Decomposition
slug: query-decomposition
date: 2026-04-20
updated: 2026-04-20
aliases: [question decomposition, query breakdown]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query decomposition** — the process of breaking a complex input question into simpler sub-queries that can be answered step by step.

## Key Points

- IterDRAG uses query decomposition to address the compositionality gap in multi-hop QA.
- Each generated sub-query triggers additional retrieval, so decomposition directly changes both evidence acquisition and compute allocation.
- The paper learns decomposition behavior from demonstrations formatted with retrieved documents, sub-queries, intermediate answers, and final answers.
- Better decomposition improves retrieval quality because simpler sub-queries are easier to match against the corpus than the original multi-hop question.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-inference]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-inference]].
