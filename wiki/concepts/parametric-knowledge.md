---
type: concept
title: Parametric Knowledge
slug: parametric-knowledge
date: 2026-04-20
updated: 2026-04-20
aliases: [model-internal knowledge, 参数化知识]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parametric Knowledge** (参数化知识) — knowledge encoded in a model's learned parameters rather than stored in an external retrieval system.

## Key Points

- The paper treats the large language model's pre-trained weights as a source of parametric knowledge that can generate useful pseudo-documents.
- Parametric knowledge alone is described as sometimes partial or insufficient for knowledge-intensive QA.
- ITRG uses generation steps to expose this internal knowledge so it can guide later retrieval.
- The framework is motivated by combining parametric knowledge with retrieved non-parametric evidence rather than choosing only one source.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[feng-2023-retrievalgeneration-2310-05149]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[feng-2023-retrievalgeneration-2310-05149]].
