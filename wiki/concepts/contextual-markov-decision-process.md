---
type: concept
title: Contextual Markov Decision Process
slug: contextual-markov-decision-process
date: 2026-04-20
updated: 2026-04-20
aliases: [CMDP, 上下文马尔可夫决策过程]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Contextual Markov Decision Process** (上下文马尔可夫决策过程) — a family of MDPs indexed by context, where the context specifies task conditions such as the goal while some underlying structure, such as dynamics, may be shared.

## Key Points

- WorldCoder formulates tasks as a CMDP `(C, S, A, M)` with natural-language goals as contexts `c \in C`.
- Each context-conditioned MDP `M(c)` shares the transition function `T` but uses a context-specific reward function `R^c`.
- This formulation lets the system transfer a learned world model across goals while only updating the reward side for a new instruction.
- The paper assumes deterministic episodic CMDPs whose episodes terminate on goal completion.
- Treating goals as contexts makes instruction following part of the same formalism as environment-model learning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-worldcoder-2402-12275]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-worldcoder-2402-12275]].
