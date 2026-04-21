---
type: concept
title: Empirical Bayes
slug: empirical-bayes
date: 2026-04-20
updated: 2026-04-20
aliases: [empirical Bayes method, 经验贝叶斯]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Empirical Bayes** (经验贝叶斯) — a Bayesian estimation strategy that uses data to estimate prior or latent-variable quantities instead of specifying them fully in advance.

## Key Points

- [[wang-2024-large-2301-11916]] uses an empirical-Bayes framing to approximate the optimal latent concept value with a small LM rather than explicitly modeling the full latent distribution.
- The paper learns token embeddings that stand in for the task concept and then scores demonstrations by how well they predict those tokens.
- This move makes the latent-variable theory computationally practical for real LLM prompting pipelines.
- The empirical-Bayes approximation is central to making demonstration selection transferable across models without full model retraining.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-large-2301-11916]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-large-2301-11916]].
