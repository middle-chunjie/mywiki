---
type: concept
title: Late-Interaction Model
slug: late-interaction-model
date: 2026-04-20
updated: 2026-04-20
aliases: [late interaction model]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Late-Interaction Model** — a retrieval architecture that preserves fine-grained token-level representations until scoring time, delaying cross-text interaction until the final matching stage.

## Key Points

- The paper uses [[colbertv2]] as its retriever for ODQA, slot-filling, and language-modeling evaluations and explicitly describes it as a late-interaction model with strong generalization.
- This choice matters because InFO-RAG is evaluated under realistic retrieved evidence rather than oracle passages.
- The late-interaction retriever supplies Top-`5` passages from a Wikipedia corpus of more than `21M` passages for several task families.
- The paper's reported gains therefore combine the proposed generator-side training with a fixed strong retrieval backbone instead of jointly training the retriever.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-unsupervised-2402-18150]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-unsupervised-2402-18150]].
