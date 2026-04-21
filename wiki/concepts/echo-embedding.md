---
type: concept
title: Echo Embedding
slug: echo-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [echo embeddings]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Echo Embedding** — an embedding extraction strategy for decoder-only language models that repeats the input and pools hidden states from the second occurrence so the pooled tokens can attend to the full original sequence.

## Key Points

- The paper introduces echo embeddings as an inference-time alternative to converting causal attention into bidirectional attention.
- A typical template is `Rewrite the following paragraph: S. The rewritten paragraph: S`, where embeddings are pooled only from the repeated `S`.
- In zero-shot evaluation on MTEB with Mistral-7B-Instruct-v0.1, echo embeddings improve average score from `42.38` to `48.64`.
- The method remains effective after supervised fine-tuning, reaching `64.68` average MTEB and slightly outperforming both causal and bidirectional classical baselines.
- The main tradeoff is compute: repeating the sequence increases token cost, though a compute-matched variant remains competitive.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[springer-2024-repetition-2402-15449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[springer-2024-repetition-2402-15449]].
