---
type: concept
title: Calibrated Binarization
slug: calibrated-binarization
date: 2026-04-20
updated: 2026-04-20
aliases: [scale-aware binarization, 校准二值化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Calibrated Binarization** (校准二值化) — a binarization strategy that preserves the usable scale and semantics of continuous hidden states while converting them into binary codes.

## Key Points

- [[unknown-nd-btrbinary-2310-01329]] inserts binarization after layer normalization but before multi-head attention inside the reader encoder.
- The paper stores variance information for passage tokens so upper layers can reconstruct scale using the saved variance and layernorm weights.
- This design is motivated by pre-layernorm encoders such as T5, where raw hidden states can range from about `-500` to `+300` and make naive `tanh` annealing collapse too early.
- Training uses a straight-through estimator: the forward pass uses discrete bits while the backward pass relies on differentiable `tanh` gradients.
- The paper presents calibrated binarization as necessary for keeping binary passage states compatible with the original upper-layer reader computation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-btrbinary-2310-01329]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-btrbinary-2310-01329]].
