---
type: concept
title: Selective Prediction
slug: selective-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [prediction with a reject option, abstention, 选择性预测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Selective Prediction** (选择性预测) — a prediction regime in which a model may abstain on low-confidence inputs so accuracy on attempted cases increases.

## Key Points

- [[menick-2022-teaching-2203-11147]] operationalizes selective prediction by thresholding reward-model scores and returning `"I don't know"` below the threshold.
- The paper shows a clear coverage-quality frontier: lower attempt rate yields higher Supported&Plausible accuracy on answered questions.
- On both NaturalQuestionsFiltered and ELI5Filtered, reward-based abstention works better than using model likelihood as the rejection signal.
- The best reported operating point exceeds `90%` S&P at roughly `70%` coverage on NaturalQuestionsFiltered and exceeds `80%` at roughly `70%` coverage on ELI5Filtered.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[menick-2022-teaching-2203-11147]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[menick-2022-teaching-2203-11147]].
