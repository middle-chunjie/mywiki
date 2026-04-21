---
type: concept
title: Position Extrapolation
slug: position-extrapolation
date: 2026-04-20
updated: 2026-04-20
aliases: [positional extrapolation, length extrapolation, 位置外推]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Position Extrapolation** (位置外推) — the adaptation of a model's positional representation so it can operate at sequence lengths substantially beyond those seen in its original pretraining.

## Key Points

- The paper treats position extrapolation as a key component of long-context continued training rather than a purely inference-time trick.
- It increases the RoPE frequency base from the Llama default to `8 x 10^6` for `64K` training and `1.28 x 10^8` for `512K` training.
- Ablations show the original RoPE base performs poorly at long lengths, while larger bases materially improve downstream long-context scores.
- The authors argue that position extrapolation must be evaluated on realistic downstream tasks, not just synthetic recall or perplexity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2024-how-2410-02660]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2024-how-2410-02660]].
