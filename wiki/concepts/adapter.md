---
type: concept
title: Adapter
slug: adapter
date: 2026-04-20
updated: 2026-04-20
aliases: [adapters, 适配器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Adapter** (适配器) — a lightweight bottleneck module inserted into a pretrained network so that task adaptation can be learned by updating a small number of additional parameters.

## Key Points

- The paper builds on bottleneck adapters with the residual form `x + \sigma(xW_down)W_up`.
- Adapter parameters are the only trainable components, while the pretrained CLIP backbone remains frozen.
- The bottleneck rank `r` controls the capacity-parameter trade-off; small `r` underfits and large `r` can hurt performance.
- Adapter initialization is kept near identity to stabilize optimization during transfer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2022-crossmodal-2211-09623]].
