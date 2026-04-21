---
type: concept
title: Behavior Collapse
slug: behavior-collapse
date: 2026-04-20
updated: 2026-04-20
aliases: [collapse to non-correcting behavior, 行为坍塌]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Behavior Collapse** (行为坍塌) — a failure mode in which training converges to a degenerate strategy that achieves reward on the training set while avoiding the intended multi-step behavior.

## Key Points

- In this paper, collapse means the model learns to produce a stronger first attempt and then make only trivial or no edits at the second attempt.
- The authors observe collapse under offline SFT-style correction training and also under naive multi-turn RL.
- Stage I is designed specifically to decouple first-turn and second-turn behavior so the policy does not immediately collapse into answer-copying.
- The reward bonus in Stage II further discourages collapse by heavily penalizing correct-to-incorrect revisions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kumar-2024-training-2409-12917]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kumar-2024-training-2409-12917]].
