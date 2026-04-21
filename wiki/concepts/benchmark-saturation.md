---
type: concept
title: Benchmark Saturation
slug: benchmark-saturation
date: 2026-04-20
updated: 2026-04-20
aliases: [saturated benchmark, benchmark ceiling effect, 基准饱和]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Benchmark Saturation** (基准饱和) — the state where benchmark scores cluster near ceiling, so the benchmark stops distinguishing among strong systems or revealing meaningful failure modes.

## Key Points

- The paper argues that recent LLMs already score above `90%` on [[bbh]], making it a weak discriminator for frontier reasoning models.
- Small output spaces, shortcut-friendly patterns, short contexts, and shallow reasoning depth are identified as major causes of saturation.
- [[bbeh]] is proposed specifically to reverse saturation while preserving the broad skill coverage that made BBH useful.
- The restored headroom is large: the best general-purpose model reaches only `9.8%` adjusted harmonic mean on BBEH.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kazemi-2025-bigbench-2502-19187]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kazemi-2025-bigbench-2502-19187]].
