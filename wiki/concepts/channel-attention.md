---
type: concept
title: Channel attention
slug: channel-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [channel-wise attention, 通道注意力]
tags: [attention, vision]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Channel attention** (通道注意力) — assigning learned importance weights to feature channels so a network can emphasize or suppress different semantic components of a representation.

## Key Points

- In FRPT, channel attention computes the gating vector `W_C` used to mix original and instance-normalized features.
- The mechanism follows a squeeze-and-excitation style design with global average pooling and two bias-free fully connected layers.
- The reduction ratio in the attention bottleneck is `r = 8`.
- The paper uses channel attention to decide which channels likely encode species discrepancies versus subcategory-discriminative evidence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
