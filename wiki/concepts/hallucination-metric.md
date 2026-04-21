---
type: concept
title: Hallucination Metric
slug: hallucination-metric
date: 2026-04-20
updated: 2026-04-20
aliases: [hallucination score, RAG hallucination rate]
tags: [evaluation, rag, metrics, hallucination]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Hallucination Metric** (幻觉指标) — a keypoint-based evaluation metric that measures the fraction of ground-truth factual keypoints actively contradicted by a generated answer, defined as `Hallu(A, K) = (1/|K|) Σᵢ 𝟙[A contradicts kᵢ]`.

## Key Points

- Distinct from [[irrelevance-metric]]: a keypoint must be specifically contradicted (not merely omitted) to count as a hallucination; missing keypoints are captured by irrelevance.
- Operates jointly with [[completeness-metric]] on the same keypoint set `K`; the three scores partition the keypoint space exhaustively.
- GPT-4o achieves the lowest hallucination rate in Chinese (12.10%) among nine tested models; MiniCPM-2B-sft the highest (28.82% CN).
- Higher retrieval quality (Recall, EIR) correlates with lower hallucination, confirming that noisy retrieval is a primary driver of generation errors.
- Proposed in RAGEval as a more targeted alternative to ROUGE-L and BLEU for identifying factual inaccuracies in RAG-generated answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-rageval-2408-01262]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-rageval-2408-01262]].
