---
type: concept
title: Tool Embedding
slug: tool-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [工具嵌入, API embedding]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tool Embedding** (工具嵌入) — a vector representation of a tool or API derived from its description or structured documentation for retrieval, ranking, or planning.

## Key Points

- TGR starts from tool embeddings produced by existing retrievers such as Paraphrase-MiniLM-L3-v2 and ToolBench-IR.
- For ToolBench, embeddings are computed from structured documents with tool names, descriptions, and parameters.
- For API-Bank, the paper uses tool descriptions only to reduce mismatch between query text and long tool documentation.
- Graph convolution refines tool embeddings so prerequisite information can affect ranking even when a prerequisite tool has weak direct semantic match to the query.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-tool-2508-05152]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-tool-2508-05152]].
