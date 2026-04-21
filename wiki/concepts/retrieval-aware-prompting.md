---
type: concept
title: Retrieval-Aware Prompting
slug: retrieval-aware-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval-informed prompting]
tags: [rag, prompting, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Aware Prompting** (检索感知提示) — a prompting scheme that injects retriever-derived signals into the generator input so the model can condition on both document text and retrieval metadata.

## Key Points

- [[ye-2024-rag-2406-13249]] projects R2-Former outputs into the LLM token-embedding space before generation.
- Each retrieved document receives a dedicated retrieval-information embedding prepended ahead of its token sequence.
- The prompt template uses `<R>` placeholders to preserve document order while marking where retrieval information should be inserted.
- The method reduces the burden on the LLM to infer which documents matter purely from long concatenated text.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2024-rag-2406-13249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2024-rag-2406-13249]].
