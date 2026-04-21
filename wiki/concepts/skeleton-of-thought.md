---
type: concept
title: Skeleton-of-Thought
slug: skeleton-of-thought
date: 2026-04-20
updated: 2026-04-20
aliases: [SoT]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Skeleton-of-Thought** — a prompting strategy that first elicits a short answer outline and then expands each outline point in parallel to reduce large-language-model inference latency.

## Key Points

- The method decomposes generation into a skeleton stage and a point-expanding stage.
- It relies on the model itself to propose a `3-10` point structure whose items can be elaborated independently.
- The same abstraction supports both API-only models through concurrent requests and local models through batched decoding.
- The paper frames SoT as a data-level efficiency technique because it changes output organization rather than the model or system stack.
- A routed version, SoT-R, uses either GPT-4 prompting or a trained RoBERTa classifier to decide when SoT should be triggered.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ning-2024-skeletonofthought-2307-15337]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ning-2024-skeletonofthought-2307-15337]].
