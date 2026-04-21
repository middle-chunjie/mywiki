---
type: concept
title: Context Discovery
slug: context-discovery
date: 2026-04-20
updated: 2026-04-20
aliases: [environment exploration, 上下文发现]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Discovery** (上下文发现) — the process of querying or probing an environment to uncover task-relevant state before attempting a final solution.

## Key Points

- [[yang-2023-intercode-2306-14898]] identifies context discovery as a central behavior in successful interactive SQL and Bash trajectories.
- In SQL, models often improve after commands such as `SHOW TABLES` or `DESC` reveal missing schema information.
- In Bash, context discovery includes inspecting files, directories, and intermediate outputs before composing a final command chain.
- The paper argues that interactive benchmarks are valuable partly because they measure this exploratory capability, which static generation benchmarks miss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-intercode-2306-14898]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-intercode-2306-14898]].
