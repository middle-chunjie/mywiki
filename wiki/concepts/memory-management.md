---
type: concept
title: Memory Management
slug: memory-management
date: 2026-04-20
updated: 2026-04-20
aliases: [memory control, 记忆管理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Memory Management** (记忆管理) — the process of deciding what information to add, update, delete, or retain in an external memory store so later retrieval remains accurate and useful.

## Key Points

- Memory-R1 frames memory management as a policy over `ADD`, `UPDATE`, `DELETE`, and `NOOP` rather than a fixed heuristic.
- The Memory Manager is optimized by downstream answer correctness, so a memory action is rewarded only if it improves later QA performance.
- The paper argues that update decisions are especially important for consolidating complementary facts instead of fragmenting memory into redundant entries.
- Retrieved old memories are part of the decision context, so memory management depends on both the new fact and the existing bank state.
- Ablation results show RL-trained memory management improves answer quality over an in-context manager baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yan-2026-memoryr-2508-19828]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yan-2026-memoryr-2508-19828]].
