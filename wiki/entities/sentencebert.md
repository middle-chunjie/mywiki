---
type: entity
title: SentenceBERT
slug: sentencebert
date: 2026-04-20
entity_type: model
aliases: [Sentence-BERT, SBERT]
tags: []
---

## Description

SentenceBERT is the pretrained text encoder used in [[he-2024-gretriever-2402-07630]] to embed query, node, and edge text before retrieval. It provides the vector space in which G-Retriever performs nearest-neighbor lookup over textual graph elements.

## Key Contributions

- Produces the node and edge representations used for indexing.
- Encodes questions into the same space so cosine-similarity retrieval can rank relevant nodes and edges.

## Related Concepts

- [[nearest-neighbor-retrieval]]
- [[retrieval-augmented-generation]]
- [[text-attributed-graph]]

## Sources

- [[he-2024-gretriever-2402-07630]]
