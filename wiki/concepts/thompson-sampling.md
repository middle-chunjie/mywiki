---
type: concept
title: Thompson Sampling
slug: thompson-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [posterior sampling, 汤普森采样]
tags: [bandits, bayesian-optimization]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Thompson Sampling** (汤普森采样) — a Bayesian bandit strategy that samples from each arm's posterior and chooses the arm that looks best under the sampled parameters.

## Key Points

- [[tang-2024-code-2405-17503]] uses Thompson Sampling to choose which partial program to refine next during LLM code repair.
- The paper models each program as a Bernoulli arm with success probability `theta_rho`, where success means one refinement yields a fully correct program.
- The posterior is `Beta(1 + C * h(rho), 1 + C * (1 - h(rho)) + N_rho)`, combining heuristic quality and the number of failed refinements.
- This lets the system exploit promising programs while still preserving nonzero probability of exploring less-tried branches.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-code-2405-17503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-code-2405-17503]].
