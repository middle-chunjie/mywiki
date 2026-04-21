---
type: concept
title: Persistent Memory
slug: persistent-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [persistent memory, 持久记忆]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Persistent Memory** (持久记忆) — memory that survives beyond a single prompt or session and can be incrementally updated as new evidence arrives.

## Key Points

- The paper distinguishes persistent memory from merely extending the context window: the goal is durable storage, not longer prompts.
- Mem0 updates persistent memory with four operations: `ADD`, `UPDATE`, `DELETE`, and `NOOP`.
- Contradictory facts are handled through explicit update logic rather than silent accumulation of duplicate memories.
- Persistent memory is motivated by personalization use cases such as retaining dietary preferences or prior commitments across sessions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chhikara-nd-mem]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chhikara-nd-mem]].
