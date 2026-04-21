---
type: concept
title: Continuous Dynamical System
slug: continuous-dynamical-system
date: 2026-04-20
updated: 2026-04-20
aliases: [continuous dynamics, 连续动力系统]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Continuous Dynamical System** (连续动力系统) — a model in which a state evolves over continuous time according to deterministic transition dynamics rather than discrete step-specific parameters.

## Key Points

- FLOATER treats position vectors as a trajectory `p(t)` driven by a latent force function `h(τ, p(τ); θ_h)`.
- The model assumes positional information changes smoothly across token indices and uses equidistant samples `t_i = i·Δ` for text.
- Continuous dynamics let the encoder extrapolate to longer sequences without storing a separate vector for each possible absolute position.
- The paper notes that the same framework could handle non-equidistant observations, such as hierarchical text or irregular time-series events.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2020-encode-2003-09229]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2020-encode-2003-09229]].
