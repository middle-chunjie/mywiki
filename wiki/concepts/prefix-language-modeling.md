---
type: concept
title: Prefix Language Modeling
slug: prefix-language-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [prefix LM, 前缀语言建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prefix Language Modeling** (前缀语言建模) — an autoregressive training setup in which a model predicts a target continuation conditioned on a prefix that may include context, prompts, or retrieved evidence.

## Key Points

- InFO-RAG keeps the decoder-only training form consistent with LLaMA pre-training by predicting `s_l^t` from `[R(s_l^p); s_l^p]`.
- The paper argues that preserving prefix LM helps maintain generality and reduces catastrophic forgetting compared with narrow supervised RAG fine-tuning.
- All three unsupervised tasks are phrased as variants of the same prefix-LM objective, differing only in how retrieved context is constructed or corrupted.
- The authors use this consistency to justify why InFO-RAG can help diverse zero-shot tasks without requiring task-specific output formats.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-unsupervised-2402-18150]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-unsupervised-2402-18150]].
