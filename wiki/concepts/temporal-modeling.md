---
type: concept
title: Temporal Modeling
slug: temporal-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [temporal context modeling, 时序建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Temporal Modeling** (时序建模) — representing and reasoning over dependencies across time steps or frames so that a model captures dynamic structure rather than isolated observations.

## Key Points

- MV-Adapter adds a dedicated temporal adaptation module to inject video dynamics into a CLIP vision encoder originally trained on images.
- The model concatenates frame-level `[CLS]` tokens, average patch features, and a learnable `[CC]` token, then applies a lightweight transformer across frames.
- It further calibrates frame-specific upsample weights from global video context and local frame context.
- Ablations show temporal modeling is necessary: a plain bottleneck branch underperforms, while temporal modeling recovers and surpasses full fine-tuning on MSR-VTT.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-mvadapter-2301-07868]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-mvadapter-2301-07868]].
