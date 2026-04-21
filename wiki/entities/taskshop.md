---
type: entity
title: TASKSHOP
slug: taskshop
date: 2026-04-20
entity_type: tool
aliases: [TaskShop]
tags: []
---

## Description

TASKSHOP is the task-selection method proposed in the paper. It estimates transfer from a source task to an unseen target by combining TASKWEB scores with a few-shot target-side scorer through pivot tasks.

## Key Contributions

- Introduces a directional transfer estimator that respects the non-commutative behavior of task transfer.
- Improves source-task ranking and top-`k` helpful-task identification over RoE and LLM-similarity baselines.
- Produces compact multi-task training sets that outperform larger baselines in zero-shot evaluation.

## Related Concepts

- [[task-selection]]
- [[pairwise-task-transfer]]
- [[multi-task-learning]]

## Sources

- [[kim-2023-taskweb-2305-13256]]
