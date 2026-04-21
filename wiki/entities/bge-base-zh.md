---
type: entity
title: BGE-base-zh-v1.5
slug: bge-base-zh
date: 2026-04-20
entity_type: tool
aliases: [BGE base zh v1.5, bge-base-zh-v1.5, C-Pack BGE]
tags: []
---

## Description

BGE-base-zh-v1.5 is a general Chinese text embedding model from the C-Pack suite (Xiao et al., 2023), developed by the BAAI/FlagEmbedding team. It encodes Chinese text into dense vectors for semantic similarity tasks.

## Key Contributions

- Used in KnowPAT as the unsupervised triple-linking retriever to encode both questions and KG triples (head, relation, tail) as text sequences.
- Enables zero-shot, unsupervised retrieval of relevant KG triples via cosine similarity, without requiring labeled question-triple pairs.

## Related Concepts

- [[dense-retrieval]]
- [[knowledge-graph-question-answering]]
- [[domain-specific-question-answering]]

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]
