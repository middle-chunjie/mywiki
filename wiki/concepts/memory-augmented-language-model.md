---
type: concept
title: Memory-Augmented Language Model
slug: memory-augmented-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [memory-augmented LM, memory LM, 记忆增强语言模型]
tags: [language-model, memory, retrieval, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Memory-Augmented Language Model** (记忆增强语言模型) — a language model that supplements its parametric weights with a non-parametric memory component (local context cache, long-range document segments, or external corpus datastore) to increase effective capacity and memorization without adding parameters.

## Key Points

- Three canonical memory types: local (preceding tokens in the same segment), long-term (previous segments of the same document beyond the attention window), and external (large corpus datastore retrieved via approximate nearest-neighbor search).
- Most prior work (continuous cache, kNN-LM) introduces memory only at inference time with a frozen base model, creating a training-inference mismatch that TRIME explicitly addresses.
- TRIME uses a contrastive training objective where in-batch examples serve as training memories, allowing gradient flow to update memory representations during training.
- Local-only memory augmentation (TRIMELM) adds negligible compute overhead while yielding consistent perplexity improvements across segment lengths.
- External memory augmentation substantially improves rare-word perplexity and enables zero-shot domain adaptation by swapping the datastore at test time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2022-training-2205-12674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2022-training-2205-12674]].
