---
type: entity
title: Pyserini
slug: pyserini
date: 2026-04-20
entity_type: tool
aliases:
  - Pyserini toolkit
tags: []
---

## Description

Pyserini is the retrieval toolkit used in [[salemi-2024-evaluating]] to run [[bm25]] and retrieve `50` documents per query for the sparse baseline experiments.

## Key Contributions

- Implements the BM25 retriever used in training and evaluation.
- Provides the sparse retrieval backend against which eRAG correlations are reported across KILT tasks.

## Related Concepts

- [[information-retrieval]]
- [[sparse-retrieval]]
- [[retrieval-evaluation]]

## Sources

- [[salemi-2024-evaluating]]
