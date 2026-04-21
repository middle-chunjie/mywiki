---
type: concept
title: Hard Negative Sampling
slug: hard-negative-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [Difficult Negative Sampling, 困难负例采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hard Negative Sampling** (困难负例采样) — the selection or construction of negative examples that are especially confusable with positives, forcing a model to learn finer discriminative cues.

## Key Points

- ConvAug creates hard negatives through entity replacement and intent shifting so conversations remain superficially similar while differing in critical semantics.
- The difficulty-adaptive filter assigns harder negatives to harder original conversations instead of using uniform random sampling.
- The paper uses `k = 1` hard negative per conversation and shows this is better than `k = 0` or `k = 2` on both in-domain and zero-shot evaluation.
- Entity replacement is the most important negative strategy in the ablation study, indicating that key-entity sensitivity is central to conversational intent understanding.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
