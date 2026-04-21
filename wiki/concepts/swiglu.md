---
type: concept
title: SwiGLU
slug: swiglu
date: 2026-04-20
updated: 2026-04-20
aliases: [Swish-Gated Linear Unit]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**SwiGLU** — a gated feed-forward activation that combines a Swish-style gate with linear projections to increase Transformer expressivity.

## Key Points

- NeoBERT swaps the GELU activation used in BERT and RoBERTa for SwiGLU.
- Because SwiGLU introduces a third weight matrix, the paper scales hidden units by `2/3` to keep overall parameter count approximately constant.
- The activation change is treated as one component of the `M1` architectural modernization ablation.
- NeoBERT aligns this choice with recent decoder families such as LLaMA, arguing that encoder-only models should inherit the same matured design patterns.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[breton-2025-neobert-2502-19587]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[breton-2025-neobert-2502-19587]].
