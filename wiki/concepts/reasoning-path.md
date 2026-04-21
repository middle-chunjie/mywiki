---
type: concept
title: Reasoning Path
slug: reasoning-path
date: 2026-04-20
updated: 2026-04-20
aliases: [reasoning trace, 推理路径]
tags: [reasoning, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reasoning Path** (推理路径) — an intermediate sequence of tokens or steps that a model generates while working toward a final answer.

## Key Points

- The paper treats each reasoning path as a latent variable `r_i` paired with a final answer `a_i`.
- Self-consistency depends on sampling diverse reasoning paths rather than only one deterministic chain of thought.
- Different reasoning paths can be lexically different yet still converge to the same correct answer, which is the basis for aggregation.
- Equation-only paths still help, but the gain is smaller than with natural-language reasoning because shorter traces leave less room for diversity.
- The observed agreement rate across sampled reasoning paths correlates with answer accuracy, suggesting a connection to uncertainty estimation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfconsistency-2203-11171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfconsistency-2203-11171]].
