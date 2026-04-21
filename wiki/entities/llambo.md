---
type: entity
title: LLAMBO
slug: llambo
date: 2026-04-20
entity_type: tool
aliases: [LLAMBO, LLMBO]
tags: []
---

## Description

LLAMBO is the LLM-augmented Bayesian optimization framework proposed in [[unknown-nd-large-2402-03921]]. It rephrases optimization state in natural language so an LLM can warm-start search, model surrogate predictions, and sample promising candidates.

## Key Contributions

- Uses zero-shot prompting to initialize BO with informative warmstart configurations.
- Implements both discriminative and generative LLM-based surrogate models over optimization history.
- Samples candidate configurations conditioned on a target score before ranking them with expected improvement.

## Related Concepts

- [[bayesian-optimization]]
- [[in-context-learning]]
- [[hyperparameter-optimization]]

## Sources

- [[unknown-nd-large-2402-03921]]
