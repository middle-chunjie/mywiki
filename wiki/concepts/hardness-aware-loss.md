---
type: concept
title: Hardness-Aware Loss
slug: hardness-aware-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [难度感知损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hardness-Aware Loss** (难度感知损失) — an objective whose gradients concentrate more strongly on difficult examples, assigning larger penalties to harder negatives than to easy ones.

## Key Points

- The paper shows that softmax contrastive loss is hardness-aware because the negative gradient magnitude is proportional to `` `exp(s_ij / tau)` ``.
- Relative negative penalties follow a Boltzmann distribution, so decreasing `tau` sharpens attention onto the hardest negatives.
- This implicit hardness weighting explains why ordinary contrastive loss outperforms a simple linear contrastive objective without hard-negative emphasis.
- The authors show that explicit hard negative sampling can substitute for the softmax-based hardness-aware mechanism.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2021-understanding-2012-09740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2021-understanding-2012-09740]].
