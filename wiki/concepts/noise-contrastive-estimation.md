---
type: concept
title: Noise-Contrastive Estimation
slug: noise-contrastive-estimation
date: 2026-04-20
updated: 2026-04-20
aliases: [NCE, 噪声对比估计]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Noise-Contrastive Estimation** (噪声对比估计) — an estimation principle that learns unnormalized models by distinguishing observed data from samples drawn from a noise or proposal distribution.

## Key Points

- CPC uses NCE-style reasoning to avoid direct modeling of the full high-dimensional observation distribution.
- The positive example is drawn from the conditional future distribution and negatives are sampled from the marginal proposal distribution.
- This lets the model estimate a useful density ratio instead of a normalized likelihood.
- The paper treats InfoNCE as the practical training objective built on top of this NCE idea.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[oord-2019-representation-1807-03748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[oord-2019-representation-1807-03748]].
