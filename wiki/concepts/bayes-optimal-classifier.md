---
type: concept
title: Bayes-Optimal Classifier
slug: bayes-optimal-classifier
date: 2026-04-20
updated: 2026-04-20
aliases: [Bayes optimal classifier, 贝叶斯最优分类器]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Bayes-Optimal Classifier** (贝叶斯最优分类器) — the classifier that minimizes expected misclassification risk under the true data-generating distribution.

## Key Points

- [[wang-2024-large-2301-11916]] proves that conditioning on the correct task latent `\theta^d` yields the Bayes-optimal classifier for the assumed causal direction.
- The paper frames demonstration selection as an attempt to make the in-context classifier approach that Bayes-optimal predictor.
- Equality with the Bayes-optimal classifier requires the posterior over demonstrations to fully concentrate on the task concept.
- This concept gives the paper its main theoretical yardstick for evaluating whether selected demonstrations are actually optimal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-large-2301-11916]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-large-2301-11916]].
