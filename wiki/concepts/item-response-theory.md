---
type: concept
title: Item Response Theory
slug: item-response-theory
date: 2026-04-20
updated: 2026-04-20
aliases: [IRT, item response model, 项目反应理论]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Item Response Theory** (项目反应理论) — a probabilistic framework that models the probability of a correct response as a function of latent examinee ability and item-specific parameters such as difficulty, discrimination, and guessing.

## Key Points

- This paper uses a three-parameter logistic IRT model with question parameters `g_i`, `d_i`, and `b_i` to evaluate RAG systems via exam responses.
- The method treats each RAG pipeline as a student and each synthetic question as an item, allowing noisy questions to contribute less than highly discriminative ones.
- A hierarchical variant decomposes model ability into `theta_llm`, `theta_ret`, and `theta_icl`, making component-level analysis possible without training a separate evaluator for each subsystem.
- The authors fit the model by maximizing joint log-likelihood with L-BFGS-B and report lower RMSE than a mean-prediction baseline on all four tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guinet-2024-automated-2405-13622]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guinet-2024-automated-2405-13622]].
