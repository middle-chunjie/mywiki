---
type: concept
title: Model-Based Repair
slug: model-based-repair
date: 2026-04-20
updated: 2026-04-20
aliases: [repair by satisfying model, 基于模型的修复]
tags: [program-analysis, program-repair]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Model-Based Repair** (基于模型的修复) - a repair strategy that uses a satisfying assignment from a constraint solver to concretize unknown program fragments.

## Key Points

- After MAX-SMT finds a satisfiable set of constraints, the paper uses the corresponding model to fill some holes automatically.
- The repair mechanism can infer declarations such as `integer` from downstream usage constraints.
- This stage goes beyond locating inconsistent nodes by actually synthesizing repaired code fragments.
- Holes that remain unresolved are sent back to the LLM in an iterative repair loop.
- The combination of solver-guided repair and LLM hole filling is a defining part of SPEAC.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mora-2024-synthetic-2406-03636]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mora-2024-synthetic-2406-03636]].
