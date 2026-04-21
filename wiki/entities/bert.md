---
type: entity
title: BERT
slug: bert
date: 2026-04-20
entity_type: model
aliases: [Bidirectional Encoder Representations from Transformers]
tags: []
---

## Description

BERT is the frozen encoder used in [[borgeaud-2022-improving-2112-04426]] to embed text chunks for nearest-neighbor retrieval. RETRO averages BERT token representations to build retrieval keys without retriever re-training.

## Key Contributions

- Provides stable chunk embeddings that let RETRO precompute retrieval indices over very large corpora.
- Removes the need to periodically refresh a jointly trained retriever during LM pre-training.

## Related Concepts

- [[dense-retrieval]]
- [[sentence-embedding]]
- [[nearest-neighbor-search]]

## Sources

- [[borgeaud-2022-improving-2112-04426]]
