---
type: concept
title: Reconstruction Loss
slug: reconstruction-loss
date: 2026-04-20
updated: 2026-04-20
aliases: [重构损失]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Reconstruction Loss** (重构损失) — the penalty measuring how well a model can reproduce its input, often used as an anomaly score when out-of-distribution examples reconstruct poorly.

## Key Points

- The paper uses cross-entropy reconstruction loss from a VAE as the main semantic quality signal for comments.
- Lower reconstruction loss indicates that a comment is more consistent with the bootstrap query corpus.
- The loss is combined with KL divergence during VAE training but ranking uses the reconstruction behavior as the anomaly signal.
- After ranking comments by this score, the method applies EM-GMM to choose the retention boundary.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-importance-2202-06649]].
