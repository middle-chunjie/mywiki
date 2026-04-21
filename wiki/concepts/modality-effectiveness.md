---
type: concept
title: Modality Effectiveness
slug: modality-effectiveness
date: 2026-04-20
updated: 2026-04-20
aliases: [unimodal effectiveness, modality quality, 模态有效性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Modality Effectiveness** (模态有效性) — a per-sample, per-modality binary signal indicating whether a unimodal representation carries sufficient task-relevant information to produce a correct unimodal prediction; used in UniS-MMC as weak supervision for contrastive pair assignment.

## Key Points

- Effectiveness is determined dynamically during training: if the unimodal classifier `g_{φ_m}` produces `ŷ_m == y` for sample `n`, modality `m` is deemed effective for that sample; otherwise ineffective.
- The signal is "weak supervision" because it is derived from the model's own predictions rather than external annotations, meaning it evolves and improves as training progresses.
- Treating effectiveness as a binary label converts the continuous "quality" spectrum of unimodal representations into an actionable contrastive objective without requiring manual quality annotations.
- The approach addresses the core failure mode of equal-treatment multimodal fusion: sensor noise, poor image quality, or ambiguous text can render one modality unreliable for a specific sample, and ignoring this degrades joint representations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zou-2023-unismmc-2305-09299]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zou-2023-unismmc-2305-09299]].
