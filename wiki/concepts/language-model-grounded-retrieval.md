---
type: concept
title: Language Model Grounded Retrieval
slug: language-model-grounded-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases:
  - LMGR
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Language Model Grounded Retrieval** — a retrieval strategy that lets a language model generate candidate concepts and then grounds them to corpus items through a retrieval or reranking stage.

## Key Points

- The paper proposes LMGR as a benchmark method tailored to long, open-domain conversations.
- LMGR has three stages: candidate generation, top-k candidate retrieval, and grounding to Wikipedia title-description pairs.
- OpenChat-3.5 is prompted to generate up to `20` candidate title-description pairs, and dense retrieval is used to find nearest Wikipedia entries.
- LMGR is strongest in the paper's reactive benchmark but less stable than ColBERT in the proactive setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[samarinas-2024-procis]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[samarinas-2024-procis]].
