---
type: concept
title: Instruction Refinement
slug: instruction-refinement
date: 2026-04-20
updated: 2026-04-20
aliases: [query refinement, prompt refinement]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Instruction Refinement** (指令改写) — the process of rewriting a user's original instruction to expose missing intent details, suppress misleading content, and improve downstream retrieval or generation.

## Key Points

- The refinement step adds unsatisfied user goals and scenario-specific tool-usage cues that are useful for the retriever.
- Refinement is conditional: if retrieved tools already satisfy the instruction and ranking is adequate, the model returns `N/A` instead of forcing a rewrite.
- The paper prefixes rewritten instructions with explicit iteration tokens so the retriever can learn stage-sensitive behavior.
- A case study shows refinement can both add important details such as "total number" and remove distracting terms such as "information."

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-enhancing-2406-17465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-enhancing-2406-17465]].
