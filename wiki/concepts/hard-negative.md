---
type: concept
title: Hard Negative
slug: hard-negative
date: 2026-04-20
updated: 2026-04-20
aliases: [困难负样本, hard negative example]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hard Negative** (困难负样本) — a negative example that is highly similar to the anchor in representation space and is therefore difficult for the model to distinguish from a positive.

## Key Points

- SCodeR mines top-ranked hard negatives from the unlabeled corpus with the current dual encoder and uses them to train the discriminators.
- The discriminators then assign graded relevance to these hard negatives instead of treating them as uniformly wrong.
- Iteratively refreshing hard negatives produces stronger supervision as the encoder improves.
- The appendix uses `negative_size = 7`, showing that hard-negative handling is part of the main training configuration rather than an auxiliary analysis only.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-softlabeled-2210-09597]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-softlabeled-2210-09597]].
