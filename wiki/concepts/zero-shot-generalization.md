---
type: concept
title: Zero-Shot Generalization
slug: zero-shot-generalization
date: 2026-04-20
updated: 2026-04-20
aliases: [Zero-Shot Transfer, 零样本泛化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Generalization** (零样本泛化) — the ability of a model trained on one data distribution to maintain useful performance on a different target distribution without additional target-task supervision.

## Key Points

- The paper evaluates zero-shot transfer by training on QReCC and testing on CAsT-20 and CAsT-21.
- ConvAug improves zero-shot retrieval to `MRR/NDCG@3 = 45.0/30.7` on CAsT-20 and `54.8/36.8` on CAsT-21.
- The gains suggest that controlled conversational augmentation helps the encoder become less tied to dataset-specific conversational forms.
- Turn-level analysis indicates larger improvements on later turns, where context complexity and distribution shift are stronger.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
