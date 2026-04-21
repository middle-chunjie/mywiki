---
type: concept
title: Long-Form Generation
slug: long-form-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [long-form text generation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Form Generation** (长文本生成) — the generation of multi-sentence or multi-paragraph responses that require coherence, completeness, and often evidence-grounded factual claims.

## Key Points

- SELF-RAG evaluates long-form biography generation with FactScore and long-form QA with ASQA metrics including MAUVE and citation precision/recall.
- The framework treats one sentence as a segment, enabling retrieval and critique to interleave with long-form output.
- Long-form tasks are where fixed-context RAG often fails because citations may not align cleanly with every generated claim.
- SELF-RAG's segment-level decoding is designed to improve support and utility across extended outputs rather than only short answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[asai-2023-selfrag-2310-11511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[asai-2023-selfrag-2310-11511]].
