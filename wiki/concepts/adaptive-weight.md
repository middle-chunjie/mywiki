---
type: concept
title: Adaptive Weight
slug: adaptive-weight
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive weighting, normalized score weight, 自适应权重]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Weight** (自适应权重) — a per-candidate normalization factor in a preference alignment loss that dynamically scales each preferred answer's gradient contribution based on its relative model-likelihood position within the preference set.

## Key Points

- KnowPAT defines `μ_i = (S_i - S_min) / (S_max - S_min)` where $S_i$ is the average log-likelihood score for answer $i$ within a preference set.
- Answers near the top of the current model distribution receive higher weights, focusing training on the most impactful preference contrasts.
- This prevents low-quality candidates from dominating the alignment gradient when the model already assigns them low probability.
- The design is motivated by the varying quality and preference degrees across different answers in both the SPS and KPS preference sets.
- Ablation shows adaptive weights provide consistent but smaller gains (removing AW: BLEU-1 −0.69 pts) compared to the preference sets themselves.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-knowledgeable-2311-06503]].
