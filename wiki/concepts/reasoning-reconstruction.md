---
type: concept
title: Reasoning Reconstruction
slug: reasoning-reconstruction
date: 2026-04-20
updated: 2026-04-20
aliases: [trace reconstruction, short-cot reconstruction]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reasoning Reconstruction** — a supervision strategy that keeps expert action traces but rewrites the intermediate reasoning into a shorter or more task-appropriate form.

## Key Points

- WebSailor does not directly imitate the verbose native thoughts of expert reasoning models.
- Instead, it retains the successful action-observation sequence and regenerates concise stepwise thoughts conditioned on the local trajectory context.
- The paper uses this design to avoid stylistic contamination while still transferring effective search strategy.
- Compact reconstructed reasoning is especially important for long-horizon web tasks because verbose trajectories quickly exhaust context windows.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-websailor-2507-02592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-websailor-2507-02592]].
