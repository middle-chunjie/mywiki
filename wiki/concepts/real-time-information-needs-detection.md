---
type: concept
title: Real-time Information Needs Detection
slug: real-time-information-needs-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [RIND, 实时信息需求检测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Real-time Information Needs Detection** (实时信息需求检测) — a token-level retrieval-triggering strategy that estimates whether an LLM currently needs external evidence by combining token uncertainty, downstream attention importance, and semantic filtering.

## Key Points

- [[su-2024-dragin-2403-10081]] defines token uncertainty with entropy over the vocabulary and token importance with the maximum last-layer attention received from future tokens.
- The method filters stopwords with a binary semantic mask so retrieval decisions focus on semantically meaningful tokens.
- RIND computes `S_RIND(t_i) = H_i * a_max(i) * s_i` and triggers retrieval when any token score exceeds threshold `theta`.
- On the IIRC ablation, the paper reports that using RIND for retrieval timing outperforms FLARE, fixed-length retrieval, and fixed-sentence retrieval for both Llama2-13B and Vicuna-13B.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2024-dragin-2403-10081]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2024-dragin-2403-10081]].
