---
type: concept
title: Partially Observable Markov Decision Process
slug: partially-observable-markov-decision-process
date: 2026-04-20
updated: 2026-04-20
aliases: [POMDP, 部分可观测马尔可夫决策过程]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Partially Observable Markov Decision Process** (部分可观测马尔可夫决策过程) — a sequential decision-making framework where the agent acts on latent states but only receives partial observations of the environment.

## Key Points

- [[yang-2023-intercode-2306-14898]] formulates interactive coding as `(\mathcal{U}, \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R})`.
- In this benchmark, code snippets are actions, execution traces are observations, and task completion is measured by a reward after `submit`.
- The formulation clarifies why exploration, memory, and adaptive replanning matter for coding agents.
- The paper uses the POMDP framing to unify Bash, SQL, Python, and prospective future tasks such as CTF under one interface.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-intercode-2306-14898]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-intercode-2306-14898]].
