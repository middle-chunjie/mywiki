---
type: concept
title: Confidence Score
slug: confidence-score
date: 2026-04-20
updated: 2026-04-20
aliases: [confidence, confidence scoring, 置信分数]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Confidence Score** (置信分数) — a scalar estimate of how certain a model is about a prediction, often used for thresholding, calibration, or selective processing.

## Key Points

- [[ma-2023-large-2303-08559]] defines sample difficulty with `` `conf(x) = max_{l ∈ L} P_SLM(l | x; s)` ``.
- Low-confidence samples are treated as hard samples and routed to an LLM reranker, while high-confidence samples are finalized directly by the SLM.
- The paper uses confidence as an unsupervised proxy for difficulty rather than training a separate hardness classifier.
- The authors explicitly note that this usage depends on the SLM being reasonably calibrated, which is a nontrivial assumption.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-large-2303-08559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-large-2303-08559]].
