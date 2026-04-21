---
type: concept
title: Keypoint Extraction
slug: keypoint-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [key point extraction, factual keypoint extraction]
tags: [evaluation, nlp, rag, metrics]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Keypoint Extraction** (关键点提取) — the process of distilling a long-form answer into a concise set of atomic factual statements (keypoints) that serve as the unit of evaluation for factual accuracy metrics such as [[completeness-metric]], [[hallucination-metric]], and [[irrelevance-metric]].

## Key Points

- Each answer is typically condensed into 3–5 keypoints, each representing an essential factual detail, inference, or conclusion required to correctly answer the question.
- Extraction is performed by an LLM using a predefined in-context learning prompt with cross-domain and cross-question-type examples, ensuring coverage of diverse answer formats.
- Keypoints enable stable, decomposable evaluation: rather than assigning a holistic score, each keypoint is independently classified as covered, contradicted, or ignored by the generated answer.
- Compared with reference-free holistic LLM scoring (e.g., RAGAS), keypoint-based evaluation yields more consistent and comparable results because the scoring space is anchored to discrete factual propositions.
- In RAGEval human validation, keypoint-level annotations achieved Fleiss' κ = 0.7686 across three independent annotators, and LLM-scored metrics differed from human scores by only 1.67% absolute.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-rageval-2408-01262]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-rageval-2408-01262]].
