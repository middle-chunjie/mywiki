---
type: concept
title: Sample Efficiency
slug: sample-efficiency
date: 2026-04-20
updated: 2026-04-20
aliases: [sample efficiency, 样本效率]
tags: [scaling, data-efficiency]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sample Efficiency** (样本效率) — the amount of learning progress a model achieves per training example or token processed.

## Key Points

- The paper argues that larger language models are more sample-efficient than smaller ones at matched loss targets.
- Compute-optimal training prefers scaling model size faster than dataset size, implying that better sample efficiency offsets reduced steps to convergence.
- The fitted overfitting relation suggests data only needs to grow sublinearly with model size, approximately `D \propto N^0.74`, to maintain similar quality.
- Appendix-B shows that deliberately training larger models for fewer updates can reduce total compute while preserving loss targets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kaplan-2020-scaling-2001-08361]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kaplan-2020-scaling-2001-08361]].
