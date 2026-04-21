---
type: concept
title: LLM2Vec
slug: llm2vec
date: 2026-04-20
updated: 2026-04-20
aliases: [LLM2Vec, llm2vec]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LLM2Vec** — a three-stage adaptation recipe that turns a decoder-only large language model into a text encoder by enabling bidirectional attention, applying masked next token prediction, and then learning pooled sentence embeddings with contrastive training.

## Key Points

- The method is explicitly designed to reuse pretrained decoder-only LLMs instead of training a separate encoder from scratch.
- Its three ingredients are bidirectional attention, MNTP adaptation, and unsupervised SimCSE.
- The paper applies the recipe to models from `1.3B` to `8B` parameters and shows gains on both token-level and sequence-level tasks.
- Mean pooling is the best sequence representation choice for LLM2Vec among the pooling variants studied.
- The strongest unsupervised result is `56.80` on full MTEB with Mistral-7B.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[behnamghader-2024-llmvec-2404-05961]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[behnamghader-2024-llmvec-2404-05961]].
