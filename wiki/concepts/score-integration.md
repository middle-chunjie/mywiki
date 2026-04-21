---
type: concept
title: Score Integration
slug: score-integration
date: 2026-04-20
updated: 2026-04-20
aliases: [mean-score integration, score aggregation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Score Integration** (分数整合) — the aggregation of multiple sampled scores for the same item into a single weighted estimate to improve reliability relative to any one sampled score.

## Key Points

- Retro* samples `K` reasoning trajectories for the same query-document pair and combines their scores with a weighted average.
- The paper uses uniform weights when generation likelihoods are unavailable, yielding the reported mean-score@`k` variants.
- During SFT data curation, the integrated score is also used as a reference to pick the teacher trajectory closest to the consensus score.
- In the RL stage, the integrated score acts as the anchor for the intra-document reward that favors stable scoring behavior.
- On BRIGHT, score integration at `k = 16` improves Retro* from `36.6` to `38.7` for 7B and from `38.5` to `40.6` for 32B.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lan-2026-retro-2509-24869]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lan-2026-retro-2509-24869]].
