---
type: concept
title: Top-k Sampling
slug: top-k-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [top k sampling, top-k 采样]
tags: [sampling, decoding]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Top-k Sampling** (top-k 采样) — a stochastic decoding method that samples the next token only from the `k` highest-probability candidates at each step.

## Key Points

- The paper uses top-`k` truncation with `k = 40` for UL2-20B, LaMDA-137B, and PaLM-540B when generating diverse reasoning paths.
- Top-k sampling is used together with temperature sampling to avoid deterministic reasoning trajectories.
- Additional ablations report that self-consistency is robust across different sampling strategies and parameter settings, including variations in `k`.
- The paper's core claim is that path diversity, not only search width, is essential for reasoning gains.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfconsistency-2203-11171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfconsistency-2203-11171]].
