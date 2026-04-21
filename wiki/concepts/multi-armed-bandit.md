---
type: concept
title: Multi-Armed Bandit
slug: multi-armed-bandit
date: 2026-04-20
updated: 2026-04-20
aliases: [MAB, contextual bandit, 多臂赌博机]
tags: [reinforcement-learning, exploration-exploitation]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Multi-Armed Bandit** (多臂赌博机) — a sequential decision framework in which an agent selects among K arms (actions) at each round to maximize cumulative reward, balancing exploitation of high-reward arms with exploration of uncertain ones.

## Key Points

- KnowGPT uses a contextual MAB to select the best combination of path extraction strategy and prompt format for each question; the context vector `c` (question embedding) conditions the arm selection.
- Each arm's expected reward is estimated as `E(Q|PF_i) = c · α_i + β_i`, where `α_i` is learned by OLS regression on past context-reward pairs and `β_i` is a UCB exploration bonus.
- The `α_i` update rule is: `α_i = ((C_i^T C_i + λ_i I)^{-1}) C_i^T r_pf_i`, a closed-form ridge regression solution updated incrementally as new LLM feedback arrives.
- The UCB term `β_i = γ * sqrt(c_i^T (C_i^T C_i + λ_i I)^{-1} c_i)` with `γ = 1 + sqrt(ln(2/δ)/2)` ensures less-tried combinations are explored.
- In the KnowGPT instantiation, the MAB manages 6 arms (2 extraction strategies × 3 prompt formats) and uses binary reward (LLM correctness on each question).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-knowgpt-2312-06185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-knowgpt-2312-06185]].
