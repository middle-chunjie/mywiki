---
type: concept
title: Feed-Forward Network
slug: feed-forward-network
date: 2026-04-20
updated: 2026-04-20
aliases: [FFN, 前馈网络]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Feed-Forward Network** (前馈网络) — the position-wise multilayer perceptron block inside a Transformer layer that transforms hidden states independently at each token position.

## Key Points

- Parametric RAG inserts document-specific LoRA updates only into FFN matrices, not into attention QKV projections.
- The paper motivates this choice by citing prior evidence that much of an LLM's factual knowledge is stored in its parameters, especially feed-forward components.
- For each retrieved document, the FFN weight is updated as `W' = W + A B^T`, and multiple documents are merged through summed low-rank updates.
- Constraining updates to FFN layers keeps the method parameter-efficient while still delivering measurable gains on QA benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2025-parametric-2501-15915]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2025-parametric-2501-15915]].
