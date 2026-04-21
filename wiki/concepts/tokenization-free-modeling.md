---
type: concept
title: Tokenization-Free Modeling
slug: tokenization-free-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [tokenizer-free modeling, 无分词建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tokenization-Free Modeling** (无分词建模) — an approach that learns directly from raw input symbols without relying on a handcrafted or learned tokenizer that maps spans into a separate token vocabulary.

## Key Points

- MEGABYTE is explicitly designed to be end-to-end differentiable over raw bytes, removing the need for BPE, SentencePiece, or image/audio tokenizers.
- The paper argues that avoiding tokenization eliminates language-specific heuristics for text and avoids lossy compression pipelines for images and audio.
- Under this architecture, tokenization-free models become competitive with strong long-context subword baselines on PG-19 while remaining general across modalities.
- The results support the paper's claim that large-scale autoregressive modeling does not require a separate tokenization stage to be viable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-megabyte-2305-07185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-megabyte-2305-07185]].
