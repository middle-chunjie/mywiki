---
type: concept
title: Uniformity-Tolerance Dilemma
slug: uniformity-tolerance-dilemma
date: 2026-04-20
updated: 2026-04-20
aliases: [均匀性-容忍度困境]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Uniformity-Tolerance Dilemma** (均匀性-容忍度困境) — the trade-off in contrastive learning between making embeddings globally uniform and preserving tolerance for semantically similar samples that should remain locally close.

## Key Points

- The paper argues that lower temperatures improve uniformity and separability but reduce tolerance to potential positives.
- Higher temperatures increase tolerance to semantically similar samples but can leave the embedding space insufficiently uniform.
- Ordinary contrastive loss therefore shows a reverse-U relation between temperature and downstream accuracy, with intermediate temperatures performing best.
- Explicit hard negative sampling alleviates the dilemma by maintaining strong uniformity while allowing larger temperatures.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2021-understanding-2012-09740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2021-understanding-2012-09740]].
