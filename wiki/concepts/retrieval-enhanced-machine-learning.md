---
type: concept
title: Retrieval-Enhanced Machine Learning
slug: retrieval-enhanced-machine-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [REML, retrieval enhanced machine learning, retrieval-augmented machine learning]
tags: [retrieval, machine-learning, information-retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Enhanced Machine Learning** (REML) — a paradigm in which ML systems are augmented with the capability to retrieve stored content at inference time, addressing limitations of purely parametric models on non-stationary data and knowledge-grounding tasks.

## Key Points

- Introduced by Zamani et al. (2022b, SIGIR) as a general framing that subsumes RAG and other retrieval-augmented approaches across different ML tasks and modalities.
- Motivates [[retrieval-augmented-generation]] as a special case: the retrieved content is used to condition text generation rather than classification or regression.
- Addresses two key failure modes of parametric-only models: inability to access knowledge not encoded during training, and poor handling of non-stationary or frequently updated information.
- End-to-end optimization within REML is inherently challenging because the retrieval step (discrete top-k selection) is non-differentiable; [[zamani-2024-stochastic]] directly addresses this for the RAG case.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zamani-2024-stochastic]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zamani-2024-stochastic]].
