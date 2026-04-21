---
type: concept
title: Long-Context Training
slug: long-context-training
date: 2026-04-20
updated: 2026-04-20
aliases: [long context training, 长上下文训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Context Training** (长上下文训练) — a training or adaptation stage that increases the context window a model can attend to while maintaining useful performance on long-document tasks.

## Key Points

- Phi-4 midtrains from `4K` to `16K` context using `250B` tokens, a `10x` lower peak learning rate, and an increased RoPE base frequency of `250K`.
- The long-context mixture is `30%` newly curated long-context data and `70%` recall tokens from earlier pretraining.
- The report claims naturally long documents work better than artificially padded sequences for improving long-context capability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
