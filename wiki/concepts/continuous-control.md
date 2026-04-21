---
type: concept
title: Continuous Control
slug: continuous-control
date: 2026-04-20
updated: 2026-04-20
aliases: [continuous control, 连续控制]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Continuous Control** (连续控制) — the ability to smoothly vary a generation attribute by changing a continuous steering parameter rather than selecting only discrete labels.

## Key Points

- LM-Switch uses the scalar `ε` as a continuous knob for the intensity and polarity of conditioning.
- Theorem 2 provides an approximate linear interpolation guarantee when moving from `0` to `ε` and then to `kε`.
- The paper visualizes sentiment shifts over `ε ∈ [-5ε_0, 5ε_0]`, showing gradual movement in output distributions rather than abrupt class switches.
- Continuous control is one reason the authors argue LM-Switch can cover nuanced scenarios without retraining separate models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
