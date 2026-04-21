---
type: concept
title: Citation Accuracy
slug: citation-accuracy
date: 2026-04-20
updated: 2026-04-20
aliases: [attribution accuracy, citation faithfulness]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Citation Accuracy** (引用准确性) — the degree to which citations attached to generated claims actually support those claims.

## Key Points

- SELF-RAG explicitly optimizes for supported citations by scoring segment candidates with `ISSUP` reflection-token probabilities.
- The paper evaluates citation quality on ASQA using citation precision and citation recall rather than answer correctness alone.
- Increasing the decoding weight on support-oriented critique tokens improves citation precision but can reduce MAUVE fluency.
- Citation accuracy is treated as a controllable property at inference time, not only as a post-hoc evaluation metric.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[asai-2023-selfrag-2310-11511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[asai-2023-selfrag-2310-11511]].
