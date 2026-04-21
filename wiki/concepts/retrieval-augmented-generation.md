---
type: concept
title: Retrieval-Augmented Generation
slug: retrieval-augmented-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [RAG, retrieval augmented generation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Augmented Generation** (检索增强生成) — a generation paradigm that conditions model outputs on external retrieved evidence in order to improve grounding, factuality, or coverage.

## Key Points

- SELF-RAG treats fixed-top-`k` RAG as a strong but blunt baseline that often retrieves even when factual grounding is unnecessary.
- The paper argues conventional RAG does not explicitly train the generator to judge whether retrieved passages are relevant or support the produced claim.
- SELF-RAG reframes RAG as a conditional behavior by predicting retrieval-control tokens such as `Retrieve=Yes/No/Continue`.
- The method shows that retrieval-augmented generation can be made more controllable by combining retrieval with self-critique and decoding-time scoring.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[asai-2023-selfrag-2310-11511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[asai-2023-selfrag-2310-11511]].
