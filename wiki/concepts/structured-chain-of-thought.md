---
type: concept
title: Structured Chain-of-Thought
slug: structured-chain-of-thought
date: 2026-04-20
updated: 2026-04-20
aliases: [structured CoT, SCoT, structured chain of thought, 结构化思维链]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Structured Chain-of-Thought** (结构化思维链) — an intermediate reasoning representation for code generation that expresses the solution process with explicit program structures and input-output specifications rather than free-form natural-language steps.

## Key Points

- The paper defines SCoT using four components: sequence structure, branch structure, loop structure, and an input-output block.
- SCoT is intended to reduce ambiguity in ordinary CoT traces, especially around iteration scope, branch conditions, and return formats.
- The representation allows nesting between structures so the model can express more complex solution plans.
- The authors observe that many generated SCoTs are close to pseudocode, but remain more abstract than full implementation-level pseudocode.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structured-2305-06599]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structured-2305-06599]].
