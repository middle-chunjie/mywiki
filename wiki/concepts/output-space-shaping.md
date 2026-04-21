---
type: concept
title: Output Space Shaping
slug: output-space-shaping
date: 2026-04-20
updated: 2026-04-20
aliases: [output space shaping]
tags: [agents, llm, training]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Output Space Shaping** — a training strategy that broadens a model's valid response space by adding diverse correct trajectories and teacher-corrected trajectories beyond the original imitation targets.

## Key Points

- In [[gou-2024-tora-2309-17452]], output space shaping is introduced to avoid over-constraining the model to a single demonstrated tool-use path per question.
- The method samples `64` candidate trajectories per training problem from the imitation-learned model, keeps correct ones, and repairs invalid ones with a teacher model.
- The teacher model is CodeLLaMA-34B trained on TORA-CORPUS, which completes truncated incorrect trajectories from plausible prefixes.
- The paper reports average gains of `+3.4%` on GSM8k and `+4.0%` on MATH from shaping, with up to `+4.5%` when correction is added.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2024-tora-2309-17452]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2024-tora-2309-17452]].
