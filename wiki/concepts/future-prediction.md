---
type: concept
title: Future Prediction
slug: future-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [未来预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Future Prediction** (未来预测) — learning by predicting observations or representations several steps ahead so the model must capture longer-range structure rather than local redundancy.

## Key Points

- CPC predicts future latent states at horizon `k` instead of reconstructing the present input.
- The paper argues that longer-horizon prediction isolates slow, semantically meaningful factors shared across time.
- In speech, performance improves substantially when the model predicts multiple future steps, peaking around `12` latent steps.
- The same future-prediction framing is reused across audio, images, natural language, and reinforcement learning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[oord-2019-representation-1807-03748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[oord-2019-representation-1807-03748]].
