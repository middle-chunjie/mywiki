---
type: concept
title: Multilingual Dense Retrieval
slug: multilingual-dense-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [多语言稠密检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Dense Retrieval** (多语言稠密检索) — dense representation-based retrieval in which queries and passages may come from multiple languages and are matched in a shared embedding space.

## Key Points

- [[thakur-2024-leveraging-2311-05800]] argues that multilingual dense retrieval is mainly constrained by the lack of large, balanced, relevance-labeled training data outside English.
- The paper evaluates both cross-lingual and monolingual retrieval under the same synthetic-data generation framework, rather than studying only one multilingual setting.
- SWIM-X uses mT5-based dual encoders trained with contrastive loss and synthetic query-passage pairs to improve multilingual retrieval without human supervision.
- Synthetic supervision yields state-of-the-art or near-state-of-the-art performance on XOR-Retrieve and XTREME-UP while remaining below supervised upper bounds on MIRACL.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[thakur-2024-leveraging-2311-05800]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[thakur-2024-leveraging-2311-05800]].
