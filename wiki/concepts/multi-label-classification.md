---
type: concept
title: Multi-Label Classification
slug: multi-label-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [multilabel classification, multi-label classification, 多标签分类]
tags: [machine-learning, supervision]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Label Classification** (多标签分类) — a supervised learning setup where each instance may be associated with multiple labels simultaneously rather than exactly one class.

## Key Points

- [[balog-2017-deepcoder-1611-01989]] casts function prediction from input-output examples as multi-label classification over `C = 34` DSL functions.
- Each function label is trained with independent negative cross entropy, so outputs can be interpreted as marginal probabilities of function presence.
- The paper connects this design to rank loss theory, arguing that marginal probabilities are sufficient for optimizing sort-and-add style search orderings.
- This formulation lets DeepCoder learn a reusable guidance signal without generating full programs token by token.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[balog-2017-deepcoder-1611-01989]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[balog-2017-deepcoder-1611-01989]].
