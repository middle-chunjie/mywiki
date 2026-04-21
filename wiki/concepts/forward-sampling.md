---
type: concept
title: Forward Sampling
slug: forward-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [rollout sampling, 前向采样]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Forward Sampling** (前向采样) — the procedure of sampling future dialogue continuations from the current conversational state in order to estimate the downstream effect of a present action.

## Key Points

- The paper uses forward sampling to approximate the expectation in MR rather than computing full future outcomes analytically.
- A rollout window `w` limits how many future turns are sampled, making the method tractable.
- Experiments use `w = 2` and sample size `3` as the default tradeoff across all tasks.
- Ablations show that larger windows generally improve reward quality, but increase time, token usage, and simulator cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-collabllm-2502-00640]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-collabllm-2502-00640]].
