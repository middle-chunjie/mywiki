---
type: concept
title: In-Batch Negatives
slug: in-batch-negatives
date: 2026-04-20
updated: 2026-04-20
aliases: [批内负样本, batch negatives]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**In-Batch Negatives** (批内负样本) — a contrastive-learning strategy that reuses other positive targets within the same mini-batch as negative examples for each anchor.

## Key Points

- The paper relies on in-batch negatives to avoid separately encoding large negative pools for contrastive dense retrieval.
- Each anchor loss depends on the entire target set `T`, so the number of negatives grows directly with batch size.
- This dependence is exactly why ordinary `gradient accumulation` cannot emulate a true large batch: each micro-batch sees fewer in-batch negatives.
- The motivation for gradient cache is to preserve the optimization benefit of many in-batch negatives without storing the full encoder graph for the whole batch.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-scaling-2101-06983]]
- [[li-2023-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-scaling-2101-06983]].
