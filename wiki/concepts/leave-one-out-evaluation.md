---
type: concept
title: Leave-One-Out Evaluation
slug: leave-one-out-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [留一评估, leave-one-out setup]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Leave-One-Out Evaluation** (留一评估) — an evaluation protocol that withholds one target item from training or calibration so the system is tested on a genuinely held-out case.

## Key Points

- The paper removes each target task and its pairwise transfer scores from TASKWEB when evaluating task selection for that target.
- This setup simulates the intended use case where the target task is unseen and only a few examples are available.
- Leave-one-out evaluation still permits gold ranking comparison because TASKWEB contains transfer scores for the held-out task.
- The same protocol is reused in the paper's multi-task experiments to maximize available sources while keeping the target unseen.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kim-2023-taskweb-2305-13256]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kim-2023-taskweb-2305-13256]].
