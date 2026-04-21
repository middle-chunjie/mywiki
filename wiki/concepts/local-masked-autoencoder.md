---
type: concept
title: Local Masked Autoencoder
slug: local-masked-autoencoder
date: 2026-04-20
updated: 2026-04-20
aliases: [LMAE]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Local Masked Autoencoder** (局部掩码自编码器) — a retrieval-oriented pretraining objective that reconstructs block tokens from global document state, local block summaries, and original token embeddings.

## Key Points

- Longtriever introduces LMAE to reduce reliance on supervised labels in long-document retrieval.
- The decoder conditions on both global `[DOC]` representation and local `[CLS]` block representation, not only token context.
- LMAE complements standard masked language modeling rather than replacing it.
- Ablation shows removing LMAE decreases MS MARCO Dev Doc performance from `0.329` to `0.307` MRR@100.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-longtriever]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-longtriever]].
