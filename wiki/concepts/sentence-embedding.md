---
type: concept
title: Sentence Embedding
slug: sentence-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [句子嵌入, sentence representation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sentence Embedding** (句子嵌入) — a dense vector representation of a sentence or paragraph designed to preserve semantic similarity relations in embedding space.

## Key Points

- BLANCA evaluates whether sentence and paragraph encoders can represent documentation and forum posts about code.
- Most tasks are scored by cosine-distance relationships in embedding space rather than by symbolic reasoning over code.
- The model pool includes SBERT-style encoders, BERTOverflow, CodeBERT, and Universal Sentence Encoder.
- Improvements after fine-tuning indicate that code-related text benefits from task-specific embedding adaptation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
