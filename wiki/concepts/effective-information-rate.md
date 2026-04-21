---
type: concept
title: Effective Information Rate
slug: effective-information-rate
date: 2026-04-20
updated: 2026-04-20
aliases: [EIR, effective info rate]
tags: [retrieval, evaluation, rag, metrics]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Effective Information Rate** (有效信息率, EIR) — a retrieval evaluation metric that measures the proportion of retrieved passage content that is relevant to the ground-truth references, computed as `EIR = Σᵢ|Gᵢ ∩ Rₜ| / Σⱼ|Rⱼ|` where the numerator counts words from matched ground-truth sentences and the denominator counts all retrieved words.

## Key Points

- Calculated at sentence level: for each ground-truth reference `Gᵢ`, individual sentences are matched against the retrieved set `Rₜ`; matched sentence word counts are summed.
- Complements retrieval Recall by penalizing noisy or verbose retrievals: a system that retrieves many irrelevant passages will have high Recall but low EIR.
- High EIR correlates with lower hallucination rates: in DragonBall experiments, BM25 achieves EIR 4.11% (CN) with hallucination 17.34%, while GTE-multilingual-Base shows EIR 2.94% (CN) with hallucination 28.35%.
- Not directly comparable across different TopK settings because the total retrieved token count changes; RAGEval excludes EIR from TopK comparison tables for this reason.
- Provides a precision-like complement to the recall-oriented retrieval Recall metric, forming a two-dimensional view of retrieval quality relevant for downstream generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2024-rageval-2408-01262]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2024-rageval-2408-01262]].
