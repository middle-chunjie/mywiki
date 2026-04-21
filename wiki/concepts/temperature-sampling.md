---
type: concept
title: Temperature Sampling
slug: temperature-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [temperature decoding, 温度采样]
tags: [decoding, sampling]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Temperature Sampling** (温度采样) — a decoding strategy that rescales token logits by temperature before sampling, changing output diversity and concentration.

## Key Points

- [[liu-nd-uncovering]] studies how temperature affects bias in CodeGen-6B rather than treating decoding as a fixed nuisance variable.
- The analysis sweeps `t` over `{0.1, 0.2, ..., 0.9}` and reports relatively high CBS around `0.3-0.5`.
- Bias decreases when `temperature > 0.6`, suggesting fairness-sensitive evaluation depends on decoding hyperparameters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-nd-uncovering]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-nd-uncovering]].
