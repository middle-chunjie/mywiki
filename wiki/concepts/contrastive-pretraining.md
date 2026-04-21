---
type: concept
title: Contrastive Pretraining
slug: contrastive-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [contrastive pre-training, 对比预训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Pretraining** (对比预训练) — a pretraining stage that learns representations by bringing matched text pairs closer and pushing mismatched pairs apart under a contrastive objective.

## Key Points

- The paper uses contrastive pretraining as the second stage after masked language modeling and before supervised finetuning.
- Its objective is a unidirectional InfoNCE loss from query to document with cosine similarity `s(q, d)` and temperature scaling.
- Training data spans `470M` public pairs across `29` datasets before filtering, then about `235M` pairs after consistency filtering.
- Batches are source-homogeneous so one batch is filled from a single dataset, reducing source-specific shortcut learning.
- The stage uses task prefixes such as `search_query`, `search_document`, `classification`, and `clustering` to disambiguate task semantics in the biencoder.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[nussbaum-2025-nomic-2402-01613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[nussbaum-2025-nomic-2402-01613]].
