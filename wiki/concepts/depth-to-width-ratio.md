---
type: concept
title: Depth-to-Width Ratio
slug: depth-to-width-ratio
date: 2026-04-20
updated: 2026-04-20
aliases: [depth width ratio, 深宽比]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Depth-to-Width Ratio** (深宽比) — the allocation of Transformer capacity between number of layers and hidden dimension for a fixed parameter budget.

## Key Points

- NeoBERT keeps BERT base width (`768`) but increases depth to `28` layers to target a more efficient parameter allocation.
- The paper argues that many small encoders are width-inefficient and benefit more from extra depth than from widening hidden states.
- Ablation `M7` first scales to `250M` parameters with a BERT-like ratio (`16` layers, width `1056`), and `M8` then shifts to the final `28`-layer, width-`768` design.
- This ratio is central to NeoBERT's claim of being a plug-and-play replacement for base encoders despite performing closer to larger backbones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[breton-2025-neobert-2502-19587]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[breton-2025-neobert-2502-19587]].
