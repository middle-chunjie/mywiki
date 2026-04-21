---
type: concept
title: Pairwise Task Transfer
slug: pairwise-task-transfer
date: 2026-04-20
updated: 2026-04-20
aliases: [成对任务迁移, intermediate task transfer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pairwise Task Transfer** (成对任务迁移) — a transfer-learning setup that measures how pretraining or fine-tuning on one source task changes downstream performance on a single target task.

## Key Points

- The paper operationalizes pairwise transfer by training on a source task and then fine-tuning on `1,000` target examples before comparing against a target-only baseline.
- TASKWEB records pairwise transfer for all task pairs among `22` NLP tasks over multiple architectures and adaptation methods.
- Transfer is directional in this benchmark: `A → B` and `B → A` often behave differently.
- The paper evaluates pairwise transfer with both percentage improvement and the fraction of random seeds that yield positive transfer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kim-2023-taskweb-2305-13256]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kim-2023-taskweb-2305-13256]].
