---
type: concept
title: Meta-Buffer
slug: meta-buffer
date: 2026-04-20
updated: 2026-04-20
aliases: [meta buffer, thought buffer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Meta-Buffer** — a lightweight retrieval store of high-level thought templates and their descriptions that an LLM consults before solving a new reasoning problem.

## Key Points

- The meta-buffer stores template tuples `(T_i, D_{T_i}, C_k)` combining a reusable thought template, its textual description, and a category label.
- BoT organizes templates into six coarse categories, including text comprehension, mathematical reasoning, and code programming.
- Retrieval is performed by comparing the distilled task representation against template descriptions with embedding similarity.
- The buffer is updated dynamically only when a newly distilled template is sufficiently dissimilar to existing ones under threshold `δ`.
- The paper positions the meta-buffer as an abstract retrieval database, contrasting it with conventional retrieval-augmented systems that store textual evidence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-buffer-2406-04271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-buffer-2406-04271]].
