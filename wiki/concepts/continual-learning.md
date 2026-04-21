---
type: concept
title: Continual Learning
slug: continual-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [持续学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Continual Learning** (持续学习) — the setting where a model must learn tasks or data distributions sequentially while retaining competence on earlier ones.

## Key Points

- The paper studies continual learning through a sequential 1D regression problem with five Gaussian peaks shown to the model one region at a time.
- Its main claim is that KANs inherit local plasticity from spline bases, so updating one region changes only nearby coefficients.
- In the toy experiment, KANs preserve previously learned peaks much better than MLPs, suggesting a natural mechanism for retention.
- The paper treats this as preliminary evidence and leaves realistic high-dimensional continual-learning settings for future work.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
