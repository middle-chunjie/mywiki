---
type: concept
title: Formal Theorem Proving
slug: formal-theorem-proving
date: 2026-04-20
updated: 2026-04-20
aliases: [automated theorem proving, theorem proving]
tags: [theorem-proving, formal-methods]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Formal Theorem Proving** (形式化定理证明) — the task of constructing machine-checkable proofs in a formal system so that correctness can be verified by a theorem prover.

## Key Points

- [[unknown-nd-leanstar]] models theorem proving in Lean as sequential prediction over proof states and tactics.
- The paper argues that tactic-only training underuses informal reasoning that humans rely on while proving theorems.
- Lean-STaR improves formal theorem proving by interleaving natural-language thoughts with tactic generation.
- The main evaluations are conducted on [[mini-f2f]] and a LeanDojo test split, showing consistent gains over SFT and few-shot baselines.
- Correctness is enforced by the proof assistant, which filters invalid trajectories during expert iteration and evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-leanstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-leanstar]].
