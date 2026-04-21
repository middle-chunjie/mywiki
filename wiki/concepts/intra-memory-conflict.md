---
type: concept
title: Intra-Memory Conflict
slug: intra-memory-conflict
date: 2026-04-20
updated: 2026-04-20
aliases: [internal memory conflict, 记忆内部冲突]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Intra-Memory Conflict** (记忆内部冲突) — a conflict where a model's own stored knowledge leads to inconsistent answers across semantically equivalent prompts or conditions.

## Key Points

- The survey attributes intra-memory conflict to `3` sources: bias in training corpora, stochastic decoding behavior, and side effects from [[knowledge-editing]].
- It frames the problem as a reliability issue because semantically equivalent prompts can elicit divergent outputs from the same model.
- Reviewed evidence shows that inconsistency appears both at the behavioral level and across internal layers or knowledge circuits.
- Solutions span consistency-aware fine-tuning, plug-in methods, output ensembling, and layer-based decoding interventions such as DoLa and ITI.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-knowledge-2403-08319]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-knowledge-2403-08319]].
