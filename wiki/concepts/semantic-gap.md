---
type: concept
title: Semantic Gap
slug: semantic-gap
date: 2026-04-20
updated: 2026-04-20
aliases: [retriever-generator mismatch]
tags: [rag, retrieval, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semantic Gap** (语义鸿沟) — the mismatch between the representations, objectives, and inductive biases of coupled components such as retrievers and generators, which makes their outputs hard to interpret jointly.

## Key Points

- In [[ye-2024-rag-2406-13249]], the semantic gap refers specifically to dense retrievers selecting documents with retrieval-side signals that decoder-only LLMs never directly observe.
- The paper argues that simple document concatenation leaves the LLM to infer relevance and inter-document relations from text alone.
- This mismatch makes RAG sensitive to noisy documents and long-context failures such as lost-in-the-middle behavior.
- R2AG treats the gap as an alignment problem and injects retrieval-side information rather than only filtering or compressing documents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2024-rag-2406-13249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2024-rag-2406-13249]].
