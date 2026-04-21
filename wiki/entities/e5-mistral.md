---
type: entity
title: E5-Mistral
slug: e5-mistral
date: 2026-04-20
entity_type: tool
aliases: [e5-mistral-7b-instruct]
tags: [retriever]
---

## Description

E5-Mistral is the dense retriever used as the stronger retrieval baseline in RAGChecker. The paper reports that it consistently outperforms BM25 on claim recall and context precision.

## Key Contributions

- Raises average claim recall from `74.0` to `83.5` relative to BM25.
- Raises average context precision from `52.3` to `61.8` relative to BM25.
- Improves end-to-end RAG F1 across all four tested generators.

## Related Concepts

- [[dense-retrieval]]
- [[context-precision]]
- [[claim-recall]]

## Sources

- [[ru-2024-ragchecker-2408-08067]]
