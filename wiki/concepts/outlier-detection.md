---
type: concept
title: Outlier Detection
slug: outlier-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [离群点检测]
tags: [filtering, sampling, data-curation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Outlier Detection** (离群点检测) — the identification of examples that differ substantially from the typical data distribution and are therefore likely to be unrepresentative or noisy.

## Key Points

- The paper applies outlier detection to document collections before question generation, not to the generated questions themselves.
- Outliers are defined as documents lying at the tails of the NI distribution, often operationalized via `k` standard deviations from the mean.
- Removing these documents improves the chance of sampling contexts from which small language models can generate useful retrieval questions.
- The ablation study shows that questions from NI-extreme documents have consistently worse HitsR than questions from the filtered synthetic dataset.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
