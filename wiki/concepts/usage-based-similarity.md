---
type: concept
title: Usage-Based Similarity
slug: usage-based-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [基于使用的相似性, class usage similarity]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Usage-Based Similarity** (基于使用的相似性) — a notion of semantic similarity derived from how similarly two classes are used in code, especially through shared invoked methods.

## Key Points

- The paper models class similarity using shared-method count `M` and the number of classes sharing those methods `C`.
- Each class pair is scored by Euclidean distance from the ideal point ``[max(M), min(C)]``.
- The resulting task measures whether textual documentation embeddings track code-usage similarity.
- Joint `HU` training on BERTOverflow gives the strongest reported correlation, reaching `0.61`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
