---
type: concept
title: Task Selection
slug: task-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [任务选择, source task selection]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Task Selection** (任务选择) — the process of choosing one or more source tasks that are expected to improve learning or generalization for a target task.

## Key Points

- The paper studies both single-task selection, where one helpful source task is chosen, and multi-task selection, where a top-`k` set of source tasks is assembled.
- The target task is assumed to be unseen except for a few examples, so task selection must work without direct transfer scores to the target.
- TASKSHOP uses pivot tasks and pairwise transfer structure to rank candidate sources more accurately than few-shot similarity baselines.
- Results show that carefully selected small task sets can outperform much larger indiscriminate multi-task mixtures.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kim-2023-taskweb-2305-13256]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kim-2023-taskweb-2305-13256]].
