---
type: concept
title: Easy-to-Hard Generalization
slug: easy-to-hard-generalization
date: 2026-04-20
updated: 2026-04-20
aliases: [easy to hard generalization, length generalization, difficulty generalization, 难度外推泛化]
tags: [reasoning, generalization, llm, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Easy-to-Hard Generalization** (难度外推泛化) — the ability of a model to correctly solve test problems that are strictly harder (e.g., require more steps, longer sequences, or deeper compositional structure) than any example seen during training or prompting.

## Key Points

- Standard chain-of-thought prompting fails at easy-to-hard generalization: accuracy drops sharply on last-letter-concatenation when test list length exceeds the prompt exemplars (from `84.2%` at length 4 to `31.8%` at length 12).
- Least-to-most prompting addresses this by building answers incrementally — each subproblem solution becomes part of the context for the next, allowing the model to reuse simpler outputs.
- This property is closely tied to compositional generalization: a model generalizing compositionally can recombine known primitives to handle novel, longer structures.
- Neural-symbolic architectures (e.g., stack machines, grammar induction) also target this problem for SCAN but require full training set access.
- The concept parallels curriculum learning and progressive difficulty training, but for in-context inference rather than training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-leasttomost-2205-10625]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-leasttomost-2205-10625]].
