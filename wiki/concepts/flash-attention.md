---
type: concept
title: Flash Attention
slug: flash-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [FlashAttention]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Flash Attention** — a fused attention computation strategy that improves hardware efficiency and throughput for Transformer training.

## Key Points

- Pythia uses Flash Attention during training to improve device throughput.
- The paper treats Flash Attention as one of the modern best-practice deviations from older GPT-style training recipes.
- In the revision history, the authors note that integrating Flash Attention materially increased throughput in the retrained suite.
- Its use helps make the dense checkpoint release of a large multi-scale suite computationally feasible.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[biderman-2023-pythia-2304-01373]]
- [[yuan-2025-native-2502-11089]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[biderman-2023-pythia-2304-01373]].
