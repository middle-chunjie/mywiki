---
type: entity
title: text-embeddings-ada-002
slug: text-embeddings-ada-002
date: 2026-04-20
entity_type: model
aliases: [text embeddings ada 002, ada-002]
tags: []
---

## Description

text-embeddings-ada-002 is the OpenAI embedding model inverted in [[morris-2023-text-2310-06816]] alongside GTR-base. The paper uses it to test whether a commercial black-box embedding API is similarly vulnerable to text reconstruction.

## Key Contributions

- Provides the MSMARCO `32`-token and `128`-token inversion settings used in the paper's main in-domain evaluation.
- Shows that even a hosted API embedder can yield `83.4` BLEU and `60.9%` exact recovery on short passages under iterative inversion.

## Related Concepts

- [[text-embedding]]
- [[embedding-inversion]]
- [[privacy-leakage]]

## Sources

- [[morris-2023-text-2310-06816]]
