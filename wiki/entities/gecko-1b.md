---
type: entity
title: Gecko-1B
slug: gecko-1b
date: 2026-04-20
entity_type: tool
aliases: [Gecko 1B, "Gecko-1B (en)"]
tags: []
---

## Description

Gecko-1B is the embedding model used in [[unknown-nd-inference]] to index document passages and queries for retrieval over Wikipedia passages from KILT. It functions as the retrieval backbone for both DRAG and IterDRAG experiments.

## Key Contributions

- Encodes both corpus passages and input queries for nearest-neighbor retrieval.
- Supports the paper's large-scale retrieval experiments over Wikipedia-derived evidence.
- Provides the retrieval layer whose quality directly affects downstream DRAG and IterDRAG performance.

## Related Concepts

- [[dense-retrieval]]
- [[document-embedding]]
- [[retrieval-augmented-generation]]

## Sources

- [[unknown-nd-inference]]
