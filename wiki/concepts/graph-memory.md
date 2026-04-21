---
type: concept
title: Graph Memory
slug: graph-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [graph memory, 图记忆]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Graph Memory** (图记忆) — a memory representation that stores entities as nodes and their relations as edges so that an agent can reason over structured dependencies instead of isolated text snippets.

## Key Points

- `Mem0^g` defines graph memory as a directed labeled graph ``G = (V, E, L)`` with typed entities, edge relations, embeddings, and timestamps.
- The system first extracts entities, then generates relation triples to encode structured conversational facts.
- New graph facts are merged with existing nodes by semantic similarity and conflict resolution, rather than naive append-only storage.
- The paper shows graph memory is particularly helpful for temporal reasoning, where explicit relations and timestamps matter.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chhikara-nd-mem]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chhikara-nd-mem]].
