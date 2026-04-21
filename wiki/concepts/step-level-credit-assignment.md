---
type: concept
title: Step-Level Credit Assignment
slug: step-level-credit-assignment
date: 2026-04-20
updated: 2026-04-20
aliases: [fine-grained credit assignment, per-step credit assignment, 步级信用分配]
tags: [agents, optimization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Step-Level Credit Assignment** (步级信用分配) — assigning learning signal to individual actions within a multi-turn trajectory by comparing returns from similar states rather than only from whole-episode outcomes.

## Key Points

- SLEA-RL computes discounted returns `\hat{R}_t` for each step and normalizes them within observation-cluster groups to obtain a step-level advantage term.
- The final advantage is `\hat{A} = A_{episode} + w \cdot A_{step}` with `w = 1.0`, combining coarse episode success with finer local state comparisons.
- This design distinguishes actions taken under similar observations but leading to different downstream outcomes, which is especially important under sparse terminal rewards.
- Ablation shows that removing the step-level term lowers ALFWorld success from `93.5%` to `87.8%` and WebShop success from `76.3%` to `70.8%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-slearl-2603-18079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-slearl-2603-18079]].
