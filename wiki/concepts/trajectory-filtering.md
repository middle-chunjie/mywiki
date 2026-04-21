---
type: concept
title: Trajectory Filtering
slug: trajectory-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [trace filtering, 轨迹过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Trajectory Filtering** (轨迹过滤) — the selective retention of sampled agent trajectories using validity, correctness, and quality criteria before they are used for training.

## Key Points

- WebDancer applies trajectory filtering after rejection sampling so that low-quality ReAct traces do not contaminate the SFT stage.
- The first filter removes invalid traces that fail to follow the required format under long-context generation.
- The second filter keeps only trajectories whose final answers are judged correct.
- The final quality stage removes hallucinated, repetitive, or poorly aligned trajectories and retains those with non-redundant information and coherent reasoning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-webdancer-2505-22648]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-webdancer-2505-22648]].
