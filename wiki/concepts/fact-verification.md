---
type: concept
title: Fact Verification
slug: fact-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [fact checking]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Fact Verification** (事实核验) — the task of deciding whether a claim is supported, refuted, or insufficiently grounded by available evidence.

## Key Points

- SELF-RAG is evaluated on PubHealth, a public-health fact-verification benchmark reported with accuracy.
- The paper separates factual support from general usefulness, using `ISSUP` for evidence alignment and `ISUSE` for response helpfulness.
- Reflection tokens let the model represent when a claim is fully supported, partially supported, or unsupported by retrieved evidence.
- The method improves fact-verification accuracy over non-retrieval and conventional retrieval baselines, especially in the 7B and 13B comparisons.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[asai-2023-selfrag-2310-11511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[asai-2023-selfrag-2310-11511]].
