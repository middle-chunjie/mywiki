---
type: concept
title: Context-Memory Conflict
slug: context-memory-conflict
date: 2026-04-20
updated: 2026-04-20
aliases: [context vs memory conflict, 上下文-记忆冲突]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context-Memory Conflict** (上下文-记忆冲突) — a knowledge conflict where external context supports an answer that disagrees with the model's parametric memory.

## Key Points

- The survey defines this as the discrepancy between contextual inputs such as prompts or retrieved documents and the knowledge encoded in model parameters.
- It identifies `2` primary causes: [[temporal-misalignment]] and [[misinformation-pollution]].
- Surveyed studies show no universal rule for whether LLMs prefer context or memory; coherent, convincing, and semantically aligned evidence often dominates.
- Mitigation strategies are grouped by objective, including faithfulness to context, resisting misinformation, disentangling answers by source, and improving overall factuality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-knowledge-2403-08319]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-knowledge-2403-08319]].
