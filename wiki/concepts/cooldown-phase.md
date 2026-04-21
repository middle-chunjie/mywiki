---
type: concept
title: Cooldown Phase
slug: cooldown-phase
date: 2026-04-20
updated: 2026-04-20
aliases: [training cooldown]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cooldown Phase** — a final stage of training that removes auxiliary conditioning signals and continues optimization on standard inputs so the model can operate normally at deployment time.

## Key Points

- MeCo reserves the final `10%` of pre-training tokens for metadata-free documents after an initial `90%` metadata-conditioned stage.
- The cooldown stage inherits both optimizer state and the current point on the cosine learning-rate schedule rather than restarting training.
- Training on `100%` URL-conditioned data performs poorly (`50.3` average), showing that cooldown is not optional.
- Directly mixing `90%` URL data and `10%` standard data throughout training helps less than the staged cooldown recipe (`56.4` vs `56.7` average).
- A separate ablation suggests `10%` to `20%` cooldown works best, while `30%` cooldown starts to hurt performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2025-metadata-2501-01956]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2025-metadata-2501-01956]].
