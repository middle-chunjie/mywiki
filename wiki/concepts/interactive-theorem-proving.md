---
type: concept
title: Interactive Theorem Proving
slug: interactive-theorem-proving
date: 2026-04-20
updated: 2026-04-20
aliases: [ITP, interactive proof assistant use, 交互式定理证明]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Interactive Theorem Proving** (交互式定理证明) — a paradigm in which humans or learning systems construct proofs step by step by interacting with a proof assistant through intermediate proof states and tactics.

## Key Points

- [[yang-nd-leandojo]] frames Lean as an interactive environment where proof states are observed, tactics are executed, and the proof assistant returns the next state or an error.
- The paper argues that ITP is more practical than fully automatic proving in many large formalization projects because proof search spaces are too large for purely automatic methods.
- LeanDojo exposes Lean through a gym-like interface, making ITP accessible to machine learning pipelines rather than only human users inside an editor.
- Reliable environment construction matters: the paper shows that handling namespaces correctly sharply reduces false proof failures relative to prior tooling.
- The resulting setup lets a learned model perform tactic generation and search while still receiving rigorous symbolic verification from Lean.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-nd-leandojo]]
- [[unknown-nd-leanstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-nd-leandojo]].
