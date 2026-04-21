---
type: concept
title: Last-Token Pooling
slug: last-token-pooling
date: 2026-04-20
updated: 2026-04-20
aliases: [last token pooling]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Last-Token Pooling** — an embedding extraction strategy for decoder-style models that uses the final token's hidden state as the pooled sequence representation.

## Key Points

- The paper uses last-token pooling on the final hidden layer of Qwen2.5-Coder to obtain sequence embeddings for both queries and documents.
- The authors report that last-token pooling outperforms mean pooling and latent attention pooling on average under matched training conditions.
- The approach is motivated by decoder-only architectures, where a trailing token naturally aggregates left-context information.
- In the paper's ablation, last-token pooling reaches `78.41%` overall average and `78.72%` on MTEB Code, the best averages among the tested pooling schemes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kryvosheieva-2025-efficient-2508-21290]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kryvosheieva-2025-efficient-2508-21290]].
