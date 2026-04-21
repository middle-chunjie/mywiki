---
type: concept
title: Emotional Stimuli
slug: emotional-stimuli
date: 2026-04-20
updated: 2026-04-20
aliases: [emotion prompt cues, 情绪刺激]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Emotional Stimuli** (情绪刺激) — emotionally charged phrases added to a prompt to trigger social, motivational, or self-regulatory effects in the model's generated response.

## Key Points

- The paper designs `11` emotional stimuli, labeled `EP01` to `EP11`, and appends them directly after the original prompt.
- The stimuli are grounded in three psychology-inspired families: self-monitoring, social cognitive theory, and cognitive emotion regulation.
- Different stimuli work best on different benchmarks: `EP02` performs best on Instruction Induction, while `EP06` performs best on BIG-Bench.
- Combining multiple emotional stimuli can sometimes improve results further, but additional cues are not uniformly beneficial once a strong single stimulus is already used.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-large-2307-11760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-large-2307-11760]].
