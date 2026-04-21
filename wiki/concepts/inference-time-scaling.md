---
type: concept
title: Inference-Time Scaling
slug: inference-time-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [test-time scaling]
tags: [agents, inference]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Inference-Time Scaling** (推理时扩展) — improving task performance by spending more compute at inference time, for example by sampling multiple candidate solutions and selecting among them.

## Key Points

- The paper implements inference-time scaling by sampling multiple agent trajectories per task and ranking them with a learned verifier.
- On OpenHands with a 32B model, Best@k rises from 20.6 at `k = 1` to 29.8 at `k = 8` and 32.0 at `k = 16`.
- The observed Best@k curve is roughly linear on a log-compute scale, suggesting useful scaling behavior.
- The gap between Pass@k and Best@k shows that inference-time scaling depends strongly on verifier quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pan-2024-training-2412-21139]]
- [[feng-2025-airrag-2501-10053]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pan-2024-training-2412-21139]].
