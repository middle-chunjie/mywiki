---
type: concept
title: Local Information Retrieval
slug: local-information-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [fine-grained retrieval, 局部信息检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Local Information Retrieval** (局部信息检索) — retrieval of the specific sentence, span, or fine-grained unit inside a document that is most relevant to a query, rather than retrieving only the document as a whole.

## Key Points

- [[liu-2025-gear-2501-02772]] formalizes local retrieval as `f(q, d) -> u`, where `u` is a query-relevant unit inside a retrieved document.
- GeAR estimates local relevance from fusion-encoder cross-attention weights instead of chunking the document into separately encoded pieces.
- The paper evaluates this capability on SQuAD, NQ, TriviaQA, PAQ, and a synthetic RIR benchmark.
- Joint generation training improves local retrieval substantially over a CL-only ablation, especially on NQ and PAQ.
- The paper frames local retrieval as a path to more interpretable search results, highlighting evidence rather than exposing only a scalar score.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2025-gear-2501-02772]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2025-gear-2501-02772]].
