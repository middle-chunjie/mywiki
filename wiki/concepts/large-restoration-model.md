---
type: concept
title: Large Restoration Model
slug: large-restoration-model
date: 2026-04-20
updated: 2026-04-20
aliases: [LRM, large restoration model, 大型图像复原模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Large Restoration Model** (大型图像复原模型) — a large-scale diffusion-based image restoration model that leverages powerful generative priors from pretrained text-to-image systems for realistic reconstruction.

## Key Points

- ReFIR studies LRMs such as [[supir]] and [[seesr]], both of which inherit strong generative priors but can hallucinate details when the degradation exceeds their internal knowledge boundary.
- The paper characterizes an LRM as having a structure-oriented ControlNet stage and a texture-oriented UNet decoder stage.
- ReFIR augments LRMs without training or parameter growth by retrieving external references and modifying decoder attention.
- The method preserves the original parameter counts of the base LRMs while trading for higher memory and latency at inference time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2024-refir-2410-05601]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2024-refir-2410-05601]].
