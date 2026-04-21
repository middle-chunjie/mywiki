---
type: concept
title: Task-Oriented Dialogue
slug: task-oriented-dialogue
date: 2026-04-20
updated: 2026-04-20
aliases: [task-oriented dialog, TOD, 任务导向对话]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Task-Oriented Dialogue** (任务导向对话) — a dialogue setting where the system interacts with a user to complete concrete goals such as booking, navigation, or information lookup using structured knowledge and task constraints.

## Key Points

- This paper focuses on end-to-end task-oriented dialogue, where the model directly maps dialogue context and KB evidence to a response without explicit belief-state annotations.
- The authors argue that practical TOD increasingly favors retrieve-then-generate designs because they fit large language models and real-world KB access better than annotation-heavy pipelines.
- A central difficulty is entity discrimination: many candidate KB entries are highly similar, so generation quality can lag behind retrieval quality.
- MK-TOD improves TOD by jointly training retrieval and generation, rather than treating them as loosely coupled modules.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shen-2023-retrievalgeneration]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shen-2023-retrievalgeneration]].
