---
type: concept
title: Atomic Identifier
slug: atomic-identifier
date: 2026-04-20
updated: 2026-04-20
aliases: [Atomic ID, 原子标识符]
tags: [retrieval, decoding]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Atomic Identifier** (原子标识符) — a document identifier design that assigns each document a single dedicated output token, allowing retrieval by one-step generation over the document vocabulary.

## Key Points

- The paper treats each docid as one output token, so the decoder needs only a single decoding step and ranking comes from sorting document-token logits.
- Atomic IDs improve effectiveness at small and medium scales, but they add corpus-size-dependent parameters proportional to embedding dimension times the number of documents.
- On MSMarcoFULL, T5-Base with Atomic IDs reaches `24.2` MRR@10, better than base Naive IDs (`13.3`) and base Semantic IDs (`11.8`).
- The paper argues that part of the Atomic-ID advantage comes from extra parameter count rather than a fundamentally better inductive bias.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pradeep-2023-how]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pradeep-2023-how]].
