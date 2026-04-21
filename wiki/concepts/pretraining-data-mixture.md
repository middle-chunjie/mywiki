---
type: concept
title: Pretraining Data Mixture
slug: pretraining-data-mixture
date: 2026-04-20
updated: 2026-04-20
aliases: [data mixture, pretraining mixture, 预训练数据配比]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pretraining Data Mixture** (预训练数据配比) — the allocation of training-token budget across heterogeneous corpora such as web, synthetic, code, or licensed data during model pretraining.

## Key Points

- Phi-4 uses a final mixture of `15%` filtered web, `15%` web rewrites, `40%` synthetic data, `20%` code, and `10%` acquired sources.
- Shorter `1T`-token ablations at `7B` scale are used to choose the mixture, based on the authors' claim of high rank correlation with longer training and larger models.
- The report treats data-mixture search as a first-class optimization problem because different sources trade off reasoning gains against knowledge coverage and hallucination behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
