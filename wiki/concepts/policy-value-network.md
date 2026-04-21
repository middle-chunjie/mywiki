---
type: concept
title: Policy-Value Network
slug: policy-value-network
date: 2026-04-20
updated: 2026-04-20
aliases: [policy/value network, policy and value network]
tags: [llm, search, learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Policy-Value Network** (策略-价值网络) — a neural model that jointly predicts action priors and state values to guide search toward promising trajectories with fewer simulations.

## Key Points

- The paper's `f_theta` predicts both `P_theta(s)` and `v_theta(s)` for each state and is used inside MCTS selection and expansion.
- XoT implements `f_theta` as a shared two-layer MLP with hidden sizes `(128, 256)` and separate policy and value heads.
- The full network has only about `10^6` parameters, making it far smaller and cheaper than the LLMs used for final inference and revision.
- Training data comes from MCTS simulations in the form `(s, ε(s), v(s))`, and the model is trained jointly with the search process over `3` iterations of `10` self-play episodes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-everything-2311-04254]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-everything-2311-04254]].
