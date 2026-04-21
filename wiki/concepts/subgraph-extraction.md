---
type: concept
title: Subgraph Extraction
slug: subgraph-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [Subgraph Retrieval, 子图抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Subgraph Extraction** (子图抽取) — the process of selecting a question-relevant subgraph from a large knowledge graph so downstream reasoning can operate on a tractable search space.

## Key Points

- The paper treats subgraph extraction as the main bottleneck in IR-KGQA because noisy facts directly degrade answer reasoning quality.
- It reformulates extraction as retrieving the evidence pattern whose instantiated graph best matches the question.
- Candidate subgraphs are not built by scoring isolated facts alone; instead, the method constrains expansion through atomic adjacency consistency checks.
- The reported answer-cover rates show that improving extraction quality is strongly correlated with higher end-to-end Hits@1.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-enhancing-2402-02175]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-enhancing-2402-02175]].
