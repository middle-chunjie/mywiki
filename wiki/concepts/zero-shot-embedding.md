---
type: concept
title: Zero-Shot Embedding
slug: zero-shot-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [zero-shot embeddings, 零样本嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Zero-Shot Embedding** (零样本嵌入) — an embedding setup in which representations are extracted from a pretrained model without any additional task-specific or unsupervised fine-tuning.

## Key Points

- The paper focuses first on zero-shot embeddings to test whether decoder-only LMs can become strong embedders purely through inference-time prompting.
- Echo embeddings reach `48.64` average on MTEB with Mistral-7B-Instruct-v0.1, compared with `42.38` for classical embeddings and `43.69` for PromptEOL.
- The compute-matched zero-shot variant still scores `49.02`, showing that the method is not useful only when extra inference tokens are available.
- Zero-shot echo embeddings are reported to be relatively insensitive to the exact wording and formatting of the repeat/rewrite prompt.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[springer-2024-repetition-2402-15449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[springer-2024-repetition-2402-15449]].
