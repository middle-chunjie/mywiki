---
type: concept
title: Knowledge Preference Alignment
slug: knowledge-preference-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [KPS, knowledge preference set, 知识偏好对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Preference Alignment** (知识偏好对齐) — a preference-alignment strategy that teaches a model to selectively use retrieved knowledge by constructing preference sets that contrast answers generated with high-quality, empty, and misleading knowledge contexts.

## Key Points

- KnowPAT constructs a Knowledge Preference Set (KPS) with four answers per QA pair: gold answer > answer using top-$k$ relevant triples > answer with no triples > answer using borderline noisy triples (rank $k+1$ to $2k$).
- The critical design choice is ranking "no knowledge" above "misleading knowledge": empirical tests showed that borderline triples have high semantic similarity but cause frequent misuse, actively degrading answer quality.
- KPS is paired with Style Preference Set (SPS) to create a unified multi-task preference dataset of `2N` sets for `N` QA pairs.
- Ablation confirms KPS is the larger contributor to performance (removing KPS: BLEU-1 −6.44 pts vs. removing SPS: −4.99 pts).
- The unsupervised nature of triple retrieval makes knowledge preference alignment particularly important, as no labeled question-knowledge supervision is available.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-knowledgeable-2311-06503]].
