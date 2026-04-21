---
type: concept
title: Expected Utility Maximization
slug: expected-utility-maximization
date: 2026-04-20
updated: 2026-04-20
aliases: [expected utility maximization, RAG expected utility, utility maximization]
tags: [optimization, rag, training-objective]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Expected Utility Maximization** — a training paradigm that directly optimizes the expected value of a task-level evaluation metric (utility function) over the output and retrieval distributions, rather than using a surrogate loss such as cross-entropy.

## Key Points

- [[zamani-2024-stochastic]] defines RAG Expected Utility as `(1/n) Σ_{(x,y)∈T} Σ_{ŷ∈Y} U(y, ŷ) · p(ŷ|x; G_θ, R_φ)`, which jointly optimizes retrieval model `R_φ` and generation model `G_θ`.
- The utility function `U` can be any metric bounded in `[0,1]` with `U(y,y)=1`, including exact match, BLEU, ROUGE, or F1; this task-agnosticity is a key design choice.
- Because the output space `Y` is unbounded for free-form generation, it is approximated by a small set of beam-search candidates refreshed every `N=10,000` training steps, with gold output `y` always included.
- Pre-computing utility values for the fixed candidate set across the next `N` steps makes optimization tractable; the stochastic retrieval distribution provides the gradient signal back to the retrieval model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zamani-2024-stochastic]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zamani-2024-stochastic]].
