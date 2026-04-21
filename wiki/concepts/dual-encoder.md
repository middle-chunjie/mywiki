---
type: concept
title: Dual Encoder
slug: dual-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [bi-encoder, 双编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dual Encoder** (双编码器) — a retrieval architecture that encodes queries and documents independently into the same vector space and ranks documents by vector similarity.

## Key Points

- PROMPTAGATOR uses a shared T5-base v1.1 encoder for both query and document towers instead of late-interaction retrievers such as ColBERT v2.
- The model mean-pools the top encoder layer and projects representations into a fixed `768`-dimensional embedding space.
- Training proceeds from Contriever-style pretraining on C4 to synthetic query-document fine-tuning with cross-entropy and in-batch negatives.
- Despite using a standard `110M` dual encoder, the paper reports `47.8` average nDCG@10 on 11 BEIR tasks in the few-shot setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dai-2022-promptagator-2209-11755]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dai-2022-promptagator-2209-11755]].
