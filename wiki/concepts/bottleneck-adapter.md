---
type: concept
title: Bottleneck Adapter
slug: bottleneck-adapter
date: 2026-04-20
updated: 2026-04-20
aliases: [bottleneck adapters, 瓶颈适配器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Bottleneck Adapter** (瓶颈适配器) — a parameter-efficient adapter module that first projects features to a lower-dimensional space, transforms them, and then projects them back to the original dimension before adding the result to the backbone.

## Key Points

- MV-Adapter uses the bottleneck flow `` `Downsample -> Transformer -> Upsample` `` in both video and text branches.
- The adapter operates on frozen CLIP features and only the bottleneck branch is trained for each retrieval task.
- The downsample and upsample matrices are denoted `` `W_down ∈ R^(d × d')` `` and `` `W_up ∈ R^(d' × d)` ``.
- The paper sets the middle dimension to `` `d' = 64` `` for `ViT-B/16`, keeping parameter growth to about `2.4%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-mvadapter-2301-07868]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-mvadapter-2301-07868]].
