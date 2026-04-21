---
type: concept
title: Real-Time Editing
slug: real-time-editing
date: 2026-04-20
updated: 2026-04-20
aliases: [real time editing, live editing]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Real-Time Editing** (实时编辑) — a generation setting where users repeatedly modify an existing context and expect the model to update its predictions immediately from the edited state.

## Key Points

- The paper contrasts real-time editing with the standard static inference setting where the prompt is fixed during decoding.
- In code assistants, insertions, deletions, and multi-place edits make naive cache reuse invalid and full recomputation too slow for interactive use.
- The challenge is not only re-encoding the edited span but also reconciling how that edit changes the positions of all subsequent cached tokens.
- RepoBench-C-8k-based insertion, deletion, and edition tasks are used to simulate this setting for next-line code completion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-let-2407-03157]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-let-2407-03157]].
