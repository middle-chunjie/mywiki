---
type: entity
title: TASKWEB
slug: taskweb
date: 2026-04-20
entity_type: tool
aliases: [TaskWeb]
tags: []
---

## Description

TASKWEB is the benchmark and repository introduced by the paper to store pairwise transfer scores among NLP tasks. It spans `22` tasks, multiple model families, and several adaptation settings.

## Key Contributions

- Provides roughly `25,000` pairwise task-transfer measurements for studying transfer structure.
- Serves as the score repository that TASKSHOP uses to estimate helpful source tasks for unseen targets.
- Supports both transfer analysis and multi-task source-set construction.

## Related Concepts

- [[benchmark]]
- [[pairwise-task-transfer]]
- [[task-transferability]]

## Sources

- [[kim-2023-taskweb-2305-13256]]
