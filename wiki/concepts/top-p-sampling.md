---
type: concept
title: Top-p Sampling
slug: top-p-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [nucleus sampling, 核采样]
tags: [decoding, sampling]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Top-p Sampling** (核采样) — a decoding method that samples from the smallest token set whose cumulative probability mass exceeds a threshold `p`.

## Key Points

- [[liu-nd-uncovering]] varies `top-p` from `0.1` to `0.9` to analyze how sampling truncation changes code bias severity.
- For CodeGen-6B, CBS is reported as highest near `top-p = 0.8`.
- Bias remains nearly unchanged when `top-p` stays in the `0.1-0.3` range, indicating a non-monotonic fairness response to nucleus sampling.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-nd-uncovering]]
- [[gou-2023-diversify]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-nd-uncovering]].
