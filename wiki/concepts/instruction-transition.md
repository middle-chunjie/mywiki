---
type: concept
title: Instruction Transition
slug: instruction-transition
date: 2026-04-20
updated: 2026-04-20
aliases: [指令切换]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Instruction Transition** (指令切换) — the change of task type between consecutive dialogue turns, such as moving between chat, clarification, single-tool, and multi-tool requests.

## Key Points

- WildToolBench makes instruction transitions ubiquitous instead of keeping tasks iid across turns.
- The benchmark allows up to `3` task-type transitions within a four-task scenario.
- Accuracy consistently drops as transitions become more frequent, sometimes by as much as `30%`.
- The paper connects these failures to self-conditioning and interference from earlier dialogue history.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2026-benchmarking-2604-06185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2026-benchmarking-2604-06185]].
