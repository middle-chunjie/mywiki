---
type: concept
title: Surrogate Model
slug: surrogate-model
date: 2026-04-20
updated: 2026-04-20
aliases: [surrogate modeling, 代理模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Surrogate Model** (代理模型) — a predictive approximation of an expensive objective function that estimates performance and often uncertainty from previously observed evaluations.

## Key Points

- LLAMBO studies both discriminative and generative surrogate designs implemented through in-context learning.
- The discriminative surrogate estimates `p(s | h; D_n)` from natural-language serializations of prior configurations and scores.
- The generative surrogate recasts TPE-style scoring as estimating `p(s <= tau | h)` through probabilistic classification.
- The paper finds that better predictive accuracy does not automatically imply better uncertainty calibration, with GP retaining an advantage on calibration metrics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
