---
type: concept
title: Proof State
slug: proof-state
date: 2026-04-20
updated: 2026-04-20
aliases: [goal state, theorem state]
tags: [theorem-proving, search]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Proof State** (证明状态) — the current machine-readable representation of goals and context that conditions the next step of a formal proof.

## Key Points

- [[unknown-nd-leanstar]] treats proof states as the state space `\mathcal{S}` in its theorem-proving MDP.
- Both the SFT baseline and Lean-STaR condition generation on the current proof state before emitting a tactic.
- Retrospective rationale generation synthesizes a thought from `(state, ground-truth tactic)` pairs rather than from the tactic alone.
- Successful proof trajectories are stored as sequences of `(state, thought, tactic)` triples for expert iteration.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-leanstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-leanstar]].
