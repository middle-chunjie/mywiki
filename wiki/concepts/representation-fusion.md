---
type: concept
title: Representation Fusion
slug: representation-fusion
date: 2026-04-20
updated: 2026-04-20
aliases: [feature fusion, 表示融合]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Representation fusion** (表示融合) — a mechanism for combining multiple latent representations into a shared hidden state so that downstream computation can use both sources of information jointly.

## Key Points

- [[unknown-nd-improving-2401-02993]] fuses retrieved sentence embeddings into transformer hidden states instead of concatenating retrieved texts to the model input.
- The fusion target is the sentence-level representation, such as the `[CLS]` token in BERT-like models.
- The paper studies two fusion operators, one based on a learnable reranker and one based on dimension-wise ordered masking.
- Representation fusion is motivated as a more compute-efficient alternative to long-context retrieval augmentation for NLU tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-improving-2401-02993]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-improving-2401-02993]].
