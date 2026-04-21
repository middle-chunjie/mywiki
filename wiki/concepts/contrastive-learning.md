---
type: concept
title: Contrastive Learning
slug: contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [对比学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Learning** (对比学习) — a representation learning paradigm that pulls semantically related examples closer and pushes unrelated or adversarial examples farther apart in embedding space.

## Key Points

- ConvAug applies contrastive learning over selected pairs of positive augmented conversations and a mixture of in-batch negatives and generated hard negatives.
- The paper uses `` `phi(x) = exp(cos(x) / tau)` `` with temperature `tau` to score similarity between encoded conversations.
- Passage ranking contrastive loss and augmentation contrastive loss are trained jointly as `` `L = L_rank + alpha * L_CL` ``.
- This objective teaches the context encoder to preserve user intent across paraphrases and to separate subtle intent shifts or entity changes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
