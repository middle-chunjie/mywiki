---
type: concept
title: Word Embedding
slug: word-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [词嵌入, word representation]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Word Embedding** (词嵌入) — a dense vector representation of a token whose geometry is learned to encode task-relevant lexical relationships.

## Key Points

- K-NRM initializes `300`-dimensional word embeddings from word2vec trained on the search corpus.
- Unlike prior histogram-based rankers, K-NRM fine-tunes embeddings end-to-end using ranking supervision.
- The paper argues that relevance-oriented embedding geometry differs from general distributional similarity and is essential for strong ranking accuracy.
- Kernel-guided updates reshape embeddings so useful query-document soft matches are emphasized while many generic semantic neighbors are pushed toward noise bands.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2017-endtoend-1706-06613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2017-endtoend-1706-06613]].
