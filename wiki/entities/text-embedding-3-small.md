---
type: entity
title: text-embedding-3-small
slug: text-embedding-3-small
date: 2026-04-20
entity_type: tool
aliases: [text-emb-3-small]
tags: []
---

## Description

text-embedding-3-small is the embedding model used in [[wang-2026-ragrouterbench-2602-00296]] to encode text chunks and entities for dense retrieval and graph seed matching.

## Key Contributions

- Encodes chunk representations with embedding dimension `1536`.
- Supports FAISS indexing for NaiveRAG and entity linking for GraphRAG.
- Provides the vector space on which intrinsic-dimension, dispersion, and hubness analyses are grounded.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[hubness]]
- [[intrinsic-dimension]]

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]
