---
type: concept
title: Multiple Negatives Ranking Loss
slug: multiple-negatives-ranking-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [MNRL, multiple negatives ranking loss, 多负样本排序损失]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multiple Negatives Ranking Loss** (多负样本排序损失) — a contrastive retrieval objective that uses in-batch negatives so each query is trained to rank its positive document above other documents in the same batch.

## Key Points

- The paper formalizes MNRL as `` `CE([PCS(q_k, d_i)]_{i=1}^n, [1, ..., n])` ``, where cosine similarities over one positive and multiple in-batch negatives feed a cross-entropy objective.
- Its effectiveness depends on large batches because the loss relies on many negative passages to shape both alignment and uniformity in embedding space.
- In the long-context regime, memory usage grows with sequence length, so effective negatives can collapse from about `` `k = 128` `` at `128` tokens to about `` `k = 2` `` at `32k`.
- On `M2-BERT-32k`, the paper reports LoCoV1 score `70.4` with MNRL, far below the `94.7` achieved by orthogonal projection loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[saad-falcon-2024-benchmarking-2402-07440]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[saad-falcon-2024-benchmarking-2402-07440]].
