---
type: concept
title: Autoregressive Search Engine
slug: autoregressive-search-engine
date: 2026-04-20
updated: 2026-04-20
aliases: [ASE, DSI-style retrieval, Neural Document Indexer, Differential Search Index, 自回归搜索引擎]
tags: [retrieval, autoregressive, document-retrieval, information-retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Autoregressive Search Engine** (自回归搜索引擎) — a document retrieval paradigm in which a language model directly generates document identifiers (titles, n-grams, hierarchical cluster IDs, or URLs) for a query via autoregressive decoding, bypassing explicit similarity computation over a dense index.

## Key Points

- Encompasses multiple naming conventions used simultaneously in the literature: Differentiable Search Index (DSI), Neural Document Indexer (NDI), SEAL, and LLM-URL are all referred to as autoregressive search engines.
- DSI (Tay et al., 2022) trains a Transformer to memorize a corpus by mapping questions to document identifiers (unstructured atomic IDs, structured string IDs, or hierarchical cluster paths); SEAL uses n-gram identifiers for improved recall.
- A key advantage over dual-encoder dense retrievers is token-level cross-attention between query and the generation target, enabling deeper query-document interactions rather than single-vector similarity.
- The training cost of corpus-specific autoregressive search engines scales sharply with corpus size, making training large-scale (175B parameter) variants impractical; LLM-URL sidesteps this by using URL generation from a pretrained LLM with no additional training.
- LLMs that have internalized web document structure can act as zero-training autoregressive search engines by generating valid Wikipedia URLs as document identifiers in zero-shot or few-shot settings.
- Performance on uncommon entities degrades significantly compared to common entities, as the model's parametric knowledge skews toward frequently occurring topics.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ziems-2023-large-2305-09612]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ziems-2023-large-2305-09612]].
