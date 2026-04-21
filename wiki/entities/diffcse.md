---
type: entity
title: DiffCSE
slug: diffcse
date: 2026-04-20
entity_type: tool
aliases: [Difference-based Contrastive Learning for Sentence Embeddings]
tags: []
---

## Description

DiffCSE is a sentence embedding baseline compared against PromCSE in [[jiang-2022-improved-2203-06875]]. It is competitive on standard STS but underperforms PromCSE on the domain-shifted CxC-STS benchmark.

## Key Contributions

- Provides a strong unsupervised comparison point with average STS `78.49`.
- Scores `70.1 ± 1.1` on CxC-STS, below PromCSE's `71.2 ± 1.1`.

## Related Concepts

- [[sentence-embedding]]
- [[contrastive-learning]]
- [[domain-shift]]

## Sources

- [[jiang-2022-improved-2203-06875]]
