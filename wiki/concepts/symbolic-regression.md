---
type: concept
title: Symbolic Regression
slug: symbolic-regression
date: 2026-04-20
updated: 2026-04-20
aliases: [symbolic equation discovery, 符号回归]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Symbolic Regression** (符号回归) — the task of discovering explicit mathematical expressions from data by searching over compositions of symbolic operators and functions.

## Key Points

- The paper frames KANs as a continuous and inspectable alternative to brittle discrete symbolic-regression pipelines.
- After pruning, users can snap learned KAN activations to symbolic library functions and export formulas with SymPy, producing a controllable simplicity-versus-accuracy frontier.
- KANs are used to recover formulas for knot invariants and mobility edges, but the paper emphasizes that human inductive bias still helps simplify the final expressions.
- Unlike pure symbolic regression, KANs can still fall back to spline-based numerical approximation when the target function is not cleanly symbolic.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
