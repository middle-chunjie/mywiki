---
type: entity
title: Qwen3-Embedding
slug: qwen3-embedding
date: 2026-04-20
entity_type: tool
aliases: [Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B, Qwen3 Embedding]
tags: []
---

## Description

Qwen3-Embedding is a family of decoder-based dense embedding models from the Qwen team, available in 0.6B, 4B, and 8B parameter variants. In [[zhou-2026-retrieve-2604-04949]], the 0.6B variant serves as the primary retriever backbone for LRAT training and evaluation; the 4B and 8B variants are used during trajectory collection analysis.

## Key Contributions

- Primary retrieval backbone for LRAT experiments; LRAT-trained Qwen3-Emb-0.6B shows 15–38% relative recall improvements on BrowseComp-Plus.
- Used during trajectory generation to represent the dense-retrieval baseline alongside BM25.
- Qwen3-30B-A3B-Thinking-2507 variant serves as the LLM judge for post-browse reasoning filtering.

## Related Concepts

- [[dense-retrieval]]
- [[bi-encoder]]
- [[agentic-search]]

## Sources

- [[zhou-2026-retrieve-2604-04949]]
