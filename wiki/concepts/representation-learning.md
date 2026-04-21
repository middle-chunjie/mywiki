---
type: concept
title: Representation Learning
slug: representation-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [representation learning, 表征学习]
tags: [embeddings, transfer]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Representation Learning** (表征学习) — learning features or embeddings that capture transferable structure in data for reuse across downstream tasks.

## Key Points

- This paper treats representation learning as a deployment problem, not just a pretraining problem: the same representation should support tasks with different compute and memory budgets.
- MRL turns one `d`-dimensional representation into a nested family of useful prefixes, avoiding separate training of many low-dimensional models.
- The paper demonstrates the idea across supervised vision, vision-language contrastive learning, and masked language modeling.
- Lower-dimensional prefixes can remain competitive with independently trained baselines, especially when the representation is explicitly optimized at logarithmic granularities.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kusupati-2024-matryoshka-2205-13147]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kusupati-2024-matryoshka-2205-13147]].
