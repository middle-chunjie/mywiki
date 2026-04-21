---
type: concept
title: Orthogonal Projection Loss
slug: orthogonal-projection-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [OPL, orthogonal projection loss, 正交投影损失]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Orthogonal Projection Loss** (正交投影损失) — a pairwise embedding objective that trains positive pairs to align and negative pairs to become orthogonal, without requiring large in-batch negative sets.

## Key Points

- The paper defines OPL as `` `MSE(PCS(q_k, p_k), y)` ``, with `` `y = 1.0` `` for positive query-document pairs and `` `y = 0.0` `` for negatives.
- Because OPL works on a single query-document pair, it remains compatible with effectively `B = 1` optimization under long-context memory limits.
- The authors adopt OPL after finding prototype loss underperforms and after identifying MNRL as too batch-hungry for `32k` retrieval training.
- In the reported ablation, OPL lifts `M2-BERT-32k` from `70.4` with MNRL and `63.2` with prototype loss to `94.7` on LoCoV1.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[saad-falcon-2024-benchmarking-2402-07440]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[saad-falcon-2024-benchmarking-2402-07440]].
