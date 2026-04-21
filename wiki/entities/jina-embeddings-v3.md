---
type: entity
title: jina-embeddings-v3
slug: jina-embeddings-v3
date: 2026-04-20
entity_type: tool
aliases: [jina-embeddings-v3, Jina Embeddings v3]
tags: []
---

## Description

jina-embeddings-v3 is a multilingual text embedding model from [[jina-ai]] and is one of the three target encoders evaluated in [[xiao-2026-embedding-2602-11047]]. In that paper it provides `1024`-dimensional embeddings for testing whether the inversion decoder transfers across encoder architectures.

## Key Contributions

- Serves as one of the target embedding spaces for the conditional diffusion inversion attack.
- Reaches `76.0%` token accuracy with the paper's sequential greedy decoder.
- Helps demonstrate that the method does not depend on access to a single fixed encoder family.

## Related Concepts

- [[text-embedding]]
- [[embedding-inversion]]
- [[privacy-leakage]]

## Sources

- [[xiao-2026-embedding-2602-11047]]
