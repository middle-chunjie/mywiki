---
type: concept
title: Knowledge Insufficiency
slug: knowledge-insufficiency
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge gap, 知识不足]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Insufficiency** (知识不足) — a failure mode in which a model's internal knowledge is inadequate for a reasoning step, causing uncertainty or error propagation through the remaining chain.

## Key Points

- Search-o1 motivates itself by showing that long reasoning traces frequently contain explicit uncertainty markers such as "perhaps" when the model lacks a needed fact.
- The paper argues that one missing factual step can corrupt many later steps in science, math, and coding problems.
- Standard RAG is presented as insufficient because it retrieves only once and cannot adapt to evolving step-specific knowledge needs.
- Agentic retrieval plus document refinement is proposed as a direct response to this failure mode.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-searcho-2501-05366]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-searcho-2501-05366]].
