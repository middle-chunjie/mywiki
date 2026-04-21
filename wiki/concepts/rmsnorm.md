---
type: concept
title: RMSNorm
slug: rmsnorm
date: 2026-04-20
updated: 2026-04-20
aliases: [Root Mean Square Layer Normalization, 均方根归一化]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**RMSNorm** (均方根归一化) — a normalization scheme that rescales activations using their root-mean-square magnitude without subtracting the mean.

## Key Points

- NeoBERT replaces classical LayerNorm with RMSNorm inside a Pre-LN residual layout.
- The paper adopts RMSNorm because it preserves training stability while requiring one fewer statistic than LayerNorm.
- This normalization change is part of the paper's broader modernization of BERT-style encoder blocks toward contemporary Transformer defaults.
- RMSNorm is paired with deeper `28`-layer scaling and larger learning rates enabled by the pre-normalized architecture.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[breton-2025-neobert-2502-19587]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[breton-2025-neobert-2502-19587]].
