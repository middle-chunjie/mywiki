---
type: concept
title: Task Transferability
slug: task-transferability
date: 2026-04-20
updated: 2026-04-20
aliases: [任务可迁移性, transferability]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Task Transferability** (任务可迁移性) — the degree to which supervision from a source task is expected to improve performance on a target task.

## Key Points

- TASKWEB treats transferability as a measurable property of ordered source-target task pairs rather than an undirected similarity score.
- The paper combines average percentage gain and positive-transfer consistency across seeds to quantify transferability.
- Transferability is not reliably commutative, which makes directional scoring important for source-task ranking.
- Positive transitivity provides useful indirect evidence about transferability through pivot tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kim-2023-taskweb-2305-13256]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kim-2023-taskweb-2305-13256]].
