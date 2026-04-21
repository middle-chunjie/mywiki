---
type: concept
title: TransE
slug: transe
date: 2026-04-20
updated: 2026-04-20
aliases: [translating embeddings, 平移嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**TransE** — a knowledge graph embedding method that represents relations as translations in vector space, encouraging head-plus-relation vectors to be close to tail vectors.

## Key Points

- [[liu-2023-enhancing]] uses TransE to pretrain concept embeddings on a pruned ConceptNet subgraph containing matched concepts and their first-order neighbors.
- The pretrained embedding table is denoted as `U in R^(N_c x v)` and initialized with embedding size `v = 768`.
- These concept embeddings are later aggregated with GraphSAGE and fused with BERT token representations inside the Knowledge-aware Text Encoder.
- The paper treats TransE as a preprocessing step whose cost can be amortized across downstream training runs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-enhancing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-enhancing]].
