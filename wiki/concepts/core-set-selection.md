---
type: concept
title: Core-Set Selection
slug: core-set-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [coreset selection, 核心集选择]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Core-Set Selection** (核心集选择) — the problem of choosing a small representative subset such that a model trained on that subset performs similarly to one trained on the full dataset.

## Key Points

- The paper recasts active learning for CNNs as unlabeled core-set selection over the pool.
- Its objective is to minimize the discrepancy between empirical loss on the selected subset and empirical loss on the full dataset.
- The theory shows that, under Lipschitz assumptions, the subset's covering radius controls this core-set loss.
- A newly labeled point is only useful to the bound when it decreases the covering radius of the selected set.
- The resulting optimization is label-agnostic at selection time and depends on the learned geometry of the representation space.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sener-2018-active-1708-00489]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sener-2018-active-1708-00489]].
