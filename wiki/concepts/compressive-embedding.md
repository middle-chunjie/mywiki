---
type: concept
title: Compressive Embedding
slug: compressive-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [压缩嵌入, summary vector]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Compressive Embedding** (压缩嵌入) — a learned embedding representation that summarizes a longer input context into a shorter sequence while preserving information useful for downstream inference.

## Key Points

- FlexRAG converts retrieved text into compressive embeddings offline before the downstream reader processes the context.
- The encoder uses the same backbone family as the downstream LLM so the resulting embeddings stay closer to the model's input embedding space.
- The paper argues that intermediate representations from the first `8` layers of `LLaMA-2-7B-chat` provide a better interface than using the entire model stack.
- Compressive embeddings are designed to support arbitrary down-sampling ratios, so one encoder output can serve different runtime budgets.
- The training objective optimizes these embeddings for answer generation quality, not only for reconstructing the original retrieved text.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-lighter-2409-15699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-lighter-2409-15699]].
