---
type: concept
title: Code Embedding
slug: code-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [code embeddings, 代码嵌入]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Embedding** (代码嵌入) — a dense vector representation of code or code-related text that preserves semantic similarity for retrieval, matching, or clustering tasks.

## Key Points

- The paper builds code embeddings from autoregressive code-generation backbones instead of encoder-only retrievers.
- It targets multiple retrieval settings with the same embedding family, including natural-language-to-code, technical QA, code-to-code, code-to-comment, and code-to-completion retrieval.
- The representations are trained contrastively on query-document pairs collected from public code datasets, benchmarks, and synthetic data.
- The model uses task-specific prefixes so the produced embeddings encode the intended retrieval objective.
- Matryoshka training makes the embeddings truncatable, allowing a quality-versus-cost trade-off at serving time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kryvosheieva-2025-efficient-2508-21290]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kryvosheieva-2025-efficient-2508-21290]].
