---
type: concept
title: Vector Quantization
slug: vector-quantization
date: 2026-04-20
updated: 2026-04-20
aliases: [VQ, 向量量化]
tags: [representation-learning, clustering]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Vector Quantization** (向量量化) — a method for compressing continuous vectors into discrete codes by assigning each point to a representative codeword with minimal distortion.

## Key Points

- The paper places GDR indexing inside the broader literature on discrete representation learning and vector quantization.
- It notes that standard Euclidean VQ corresponds to `k`-means under a distortion-minimization objective.
- This connection explains why document-space hierarchical `k`-means is a strong baseline for static numeric IDs.
- BMI extends the idea by quantizing query-conditioned document representations rather than raw document embeddings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2024-bottleneckminimal-2405-10974]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2024-bottleneckminimal-2405-10974]].
