---
type: concept
title: Completeness Metric
slug: completeness-metric
date: 2026-04-20
updated: 2026-04-20
aliases: [completeness score, RAG completeness]
tags: [evaluation, rag, metrics]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Completeness Metric** (完整性指标) — a keypoint-based evaluation metric that measures the fraction of ground-truth factual keypoints semantically covered by a generated answer, defined as `Comp(A, K) = (1/|K|) Σᵢ 𝟙[A covers kᵢ]`.

## Key Points

- Operates on a set of keypoints `K = {k₁, …, kₙ}` extracted from the gold answer via LLM prompting; coverage is assessed semantically, not lexically.
- A keypoint is "covered" if the generated answer contains consistent and accurate information about it — no contradictions or factual errors permitted.
- Complements [[hallucination-metric]] and [[irrelevance-metric]]: the three scores sum to 1, partitioning keypoints into covered, contradicted, and ignored.
- Proposed in RAGEval as part of a stable, comparable scoring system that avoids the instability of direct LLM grading without reference points.
- Correlates more faithfully with human judgments than surface-level metrics: GPT-4o scores 79.13% Completeness (CN) while achieving only 21.30% Rouge-L, whereas Baichuan-2-7B achieves the reverse.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-rageval-2408-01262]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-rageval-2408-01262]].
