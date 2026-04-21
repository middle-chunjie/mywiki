---
type: concept
title: Memory Overwrite Strategy
slug: memory-overwrite-strategy
date: 2026-04-20
updated: 2026-04-20
aliases: [overwrite memory]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Memory Overwrite Strategy** — a long-context processing scheme in which a model repeatedly replaces a fixed-length memory state with a new token sequence that preserves only the most task-relevant information seen so far.

## Key Points

- MemAgent updates memory after every document chunk instead of appending to a growing context, which keeps the active window size constant.
- The overwrite operation is learned end-to-end with reinforcement learning rather than by a handcrafted compression heuristic.
- Because memory length stays fixed at `1024` tokens in the main setup, inference cost scales linearly with the number of chunks.
- The paper argues that explicit overwrite encourages aggressive retention of answer-critical facts and removal of distractors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2026-memagent-2507-02259]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2026-memagent-2507-02259]].
