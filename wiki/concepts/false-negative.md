---
type: concept
title: False Negative
slug: false-negative
date: 2026-04-20
updated: 2026-04-20
aliases: [假负样本, false negative pair]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**False Negative** (假负样本) — an example treated as a negative during training even though it is actually semantically related or equivalent to the anchor example.

## Key Points

- The paper identifies duplicated and closely related programs in large code corpora as a core source of false negatives for contrastive learning.
- SCodeR addresses this by replacing binary negative handling with discriminator-produced soft relevance scores.
- In the case study, a semantically matching negative receives a high soft label close to the true positive, showing how the method can reduce over-separation.
- The false-negative problem is especially acute for code because many independently written functions implement nearly the same behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-softlabeled-2210-09597]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-softlabeled-2210-09597]].
