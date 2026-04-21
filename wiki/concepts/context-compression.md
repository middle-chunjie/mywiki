---
type: concept
title: Context Compression
slug: context-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [上下文压缩, prompt compression]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context compression** (上下文压缩) — reducing the tokenized context presented to a language model while preserving as much task-relevant information as possible for downstream generation.

## Key Points

- xRAG compresses each retrieved passage from roughly `175.1` tokens on average to a single projected retrieval token.
- The paper contrasts xRAG with hard-prompting and soft-prompting compression methods, arguing that many prior methods either store large activation memories or achieve weaker compression.
- xRAG reuses document embeddings that already exist for dense retrieval, so compression does not require storing per-document LLM activations.
- On both Mistral-7B and Mixtral-8x7B, xRAG substantially outperforms LLMLingua and TF-IDF while keeping the compressed context length at `1`.
- The paper shows that aggressive compression works best on retrieval-supported factual QA, but loses more information on tasks requiring multi-hop reasoning over evidence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-xrag-2405-13792]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-xrag-2405-13792]].
