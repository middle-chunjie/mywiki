---
type: concept
title: NT-Xent Loss
slug: nt-xent-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [normalized temperature-scaled cross-entropy loss, NTXent]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**NT-Xent Loss** — a contrastive objective that maximizes similarity between a positive pair while normalizing against negatives through a temperature-scaled softmax.

## Key Points

- PromCSE uses NT-Xent as the base contrastive objective for both unsupervised and supervised sentence embedding learning.
- In the unsupervised setting, positives are created by two passes of the same sentence under different dropout masks, while other in-batch examples act as negatives.
- In the supervised setting, the denominator is extended to include contradiction hypotheses as hard negatives in addition to ordinary in-batch negatives.
- The paper argues that with small temperature `tau = 0.05`, NT-Xent still does not enforce a sufficient explicit margin, which motivates the auxiliary energy-based hinge loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-improved-2203-06875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-improved-2203-06875]].
