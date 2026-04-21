---
type: concept
title: Backtracking
slug: backtracking
date: 2026-04-20
updated: 2026-04-20
aliases: [回溯]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Backtracking** (回溯) — a reasoning operation in which a model abandons or revises a prior line of thought after detecting inconsistency, error, or a more promising alternative.

## Key Points

- The paper identifies backtracking as part of the structural signature of effective long-CoT demonstrations.
- Deleting or shuffling reasoning steps weakens the student model partly because it destroys the dependencies required for meaningful backtracking.
- The student can imitate backtracking-style phrases even on corrupted traces, but those surface cues alone do not recover performance.
- Correctly placed backtracking behavior is presented as one reason structurally coherent traces outperform content-correct but structurally damaged traces.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-llms-2502-07374]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-llms-2502-07374]].
