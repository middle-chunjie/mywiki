---
type: concept
title: Distractor Images
slug: distractor-images
date: 2026-04-20
updated: 2026-04-20
aliases: [negative gallery images, 干扰图像]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Distractor Images** (干扰图像) — non-relevant gallery items intentionally included in a retrieval benchmark to create challenging false positives and test embedding robustness.

## Key Points

- [[wu-2023-forb-2309-16249]] includes `49,850` distractors in addition to `4,585` index images to make flat-object retrieval meaningfully harder.
- Distractors are drawn from the same set of eight object domains and are selected to share semantics, content, or textures with true matches.
- The benchmark uses distractors to probe matching-score margin rather than relying only on top-rank correctness.
- The paper identifies broader-domain distractors as a clear future extension because current distractors are still relatively in-domain.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2023-forb-2309-16249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2023-forb-2309-16249]].
