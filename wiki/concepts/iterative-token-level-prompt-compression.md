---
type: concept
title: Iterative Token-Level Prompt Compression
slug: iterative-token-level-prompt-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [ITPC, iterative token compression, 迭代式词元级提示压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Iterative Token-Level Prompt Compression** (迭代式词元级提示压缩) — a segment-wise token filtering procedure that keeps informative tokens while conditioning each compression step on previously retained compressed content.

## Key Points

- [[jiang-2023-llmlingua-2310-05736]] introduces ITPC to relax the independence assumption behind one-shot perplexity-based token removal.
- The prompt is split into segments `S = {s_1, ..., s_m}`, and each segment is rescored conditioned on compressed prefixes `s~_{<j}`.
- Segment-specific thresholds `γ_j` are derived from the corresponding target rate for instruction, demonstrations, or question, rather than using a single threshold for the whole prompt.
- The retention rule keeps tokens whose estimated probabilities exceed `γ_j`, prioritizing high-information tokens under the local budget.
- Removing ITPC in ablation drops GSM8K `1-shot` EM from `79.08` to `72.93`, showing that conditional dependence matters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2023-llmlingua-2310-05736]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2023-llmlingua-2310-05736]].
