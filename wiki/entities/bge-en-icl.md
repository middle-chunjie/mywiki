---
type: entity
title: bge-en-icl
slug: bge-en-icl
date: 2026-04-20
entity_type: tool
aliases: [BGE-en-ICL, BGE-EN-ICL]
tags: []
---

## Description

`bge-en-icl` is the English embedding model introduced in [[li-2024-making-2409-15700]]. It is built on [[mistral-7b]] and adds query-side few-shot demonstrations so embeddings can exploit in-context learning.

## Key Contributions

- Achieves `71.67` average on full-data MTEB in the paper's few-shot setting.
- Shows that query-side ICL can outperform more invasive attention and pooling changes.
- Reaches strong out-of-distribution retrieval results on [[air-benchmark]].

## Related Concepts

- [[text-embedding]]
- [[in-context-learning]]
- [[dense-retrieval]]
- [[last-token-pooling]]

## Sources

- [[li-2024-making-2409-15700]]
