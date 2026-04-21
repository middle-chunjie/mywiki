---
type: concept
title: Logistic Regression
slug: logistic-regression
date: 2026-04-20
updated: 2026-04-20
aliases: [logit model, logistic classifier, 逻辑回归]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Logistic Regression** (逻辑回归) — a linear probabilistic classifier that maps features to binary outcome probabilities through the sigmoid function.

## Key Points

- The paper uses regularized logistic regression as the only trainable layer on top of frozen LLaMA embeddings to predict human choices.
- This setup turns high-dimensional language-model representations into a transparent choice model `σ(w^T h + b)` without modifying the base network weights.
- Model selection relies on nested cross-validation over the `ℓ_2` regularization coefficient `α`, showing that performance depends on calibrating the linear readout rather than altering the backbone.
- A random-effects extension adds participant-specific variation on top of the same logistic form, improving fit on the horizon task.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-turning-2306-03917]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-turning-2306-03917]].
