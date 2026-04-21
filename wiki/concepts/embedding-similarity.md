---
type: concept
title: Embedding Similarity
slug: embedding-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [embedding similarity, 嵌入相似度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Embedding Similarity** (嵌入相似度) — similarity between learned vector representations, here used to rank prompts by cosine closeness to a copyrighted character's name embedding.

## Key Points

- The paper's EmbeddingSim ranking computes cosine similarity between the text-encoder embedding of the character name and each candidate keyword or description.
- Higher embedding similarity is empirically associated with a greater chance of generating the intended copyrighted character.
- Among `100` descriptions per character, the top-ranked description by embedding similarity generates `24` characters successfully versus `16` for the bottom-ranked description.
- Embedding-ranked keywords are competitive with LAION co-occurrence keywords and remain useful in both attack discovery and mitigation design.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-fantastic-2406-14526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-fantastic-2406-14526]].
