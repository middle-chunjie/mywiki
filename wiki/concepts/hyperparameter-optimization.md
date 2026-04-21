---
type: concept
title: Hyperparameter Optimization
slug: hyperparameter-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [HPO, hyperparameter tuning, 超参数优化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hyperparameter Optimization** (超参数优化) — the process of searching over model hyperparameters to maximize or minimize an evaluation metric on a given task.

## Key Points

- The paper uses HPO as the main evaluation setting because LLMs may already encode priors about common model families and tabular learning tasks.
- Experiments cover public Bayesmark tasks, HPOBench tasks, and additional private or synthetic tasks to test generalization beyond memorized public datasets.
- LLAMBO frames HPO metadata, search ranges, and optimization history in natural language so an LLM can propose and score configurations.
- Gains are reported to be largest in the early stage of search, where only a handful of evaluations are available.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-large-2402-03921]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-large-2402-03921]].
