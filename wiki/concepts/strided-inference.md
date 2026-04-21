---
type: concept
title: Strided Inference
slug: strided-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [offset inference, 跨步推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Strided Inference** (跨步推理) — an inference procedure that runs a model multiple times with shifted input offsets and combines the resulting predictions to reduce position-dependent errors.

## Key Points

- MEGABYTE observes that predictions tend to worsen near the end of each patch, where the weaker local model carries more of the burden.
- The proposed fix runs two full forward passes offset by `P / 2` tokens and merges the first half of each patch from the two passes.
- On PG-19, strided inference improves the reported bpb from `0.9079` to `0.8926`, and combining it with sliding windows reaches `0.8751`.
- The paper emphasizes that the quality gain is not free: strided inference doubles inference cost on its own.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-megabyte-2305-07185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-megabyte-2305-07185]].
