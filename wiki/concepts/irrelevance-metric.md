---
type: concept
title: Irrelevance Metric
slug: irrelevance-metric
date: 2026-04-20
updated: 2026-04-20
aliases: [irrelevancy score, RAG irrelevance]
tags: [evaluation, rag, metrics]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Irrelevance Metric** (不相关指标) — a keypoint-based evaluation metric that quantifies the fraction of ground-truth factual keypoints neither covered nor contradicted by a generated answer, computed as `Irr(A, K) = 1 − Comp(A, K) − Hallu(A, K)`.

## Key Points

- Derived from [[completeness-metric]] and [[hallucination-metric]]; no independent LLM scoring call is needed — it is the residual fraction of keypoints the answer ignores.
- Captures "silent omissions" where the model fails to engage with relevant information at all, which ROUGE-L and BLEU subsume within their overlap penalty but do not distinguish from contradictions.
- GPT-4o achieves the lowest Irrelevance among evaluated models (8.77% CN, 16.85% EN), while MiniCPM-2B-sft reaches 16.58% (CN) and 22.63% (EN).
- Increasing retrieval TopK from 2 to 5 substantially reduces Irrelevance, confirming that omission is partly a retrieval coverage problem.
- Together with [[completeness-metric]] and [[hallucination-metric]], provides a complete diagnostic of answer quality: completeness addresses recall of facts, hallucination addresses factual errors, and irrelevance addresses omission.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-rageval-2408-01262]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-rageval-2408-01262]].
