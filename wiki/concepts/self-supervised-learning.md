---
type: concept
title: Self-Supervised Learning
slug: self-supervised-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [self-supervision, 自监督学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Supervised Learning** (自监督学习) — a learning paradigm that derives supervision directly from the structure of unlabeled data instead of relying on externally annotated labels.

## Key Points

- REVELA trains dense retrievers from raw corpora without annotated or synthetic query-document pairs.
- The paper instantiates self-supervision through language modeling rather than contrastive pseudo-pair generation.
- Retrieval supervision arises from whether cross-document context helps next-token prediction inside the LM.
- The method is evaluated across code, reasoning-intensive, and general retrieval tasks to argue that the resulting supervision transfers across domains.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cai-2026-revela-2506-16552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cai-2026-revela-2506-16552]].
