---
type: concept
title: Tool Retrieval
slug: tool-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [API retrieval, tool search, 工具检索]
tags: [agents, retrieval, tool-use]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Retrieval** (工具检索) — retrieving or ranking candidate tools from a large tool corpus so an agent can decide which API to invoke.

## Key Points

- ToolBench retrieval in this paper spans both in-domain and multi-domain settings, with the latter requiring search over nearly `47k` APIs.
- ToolWeaver uses a retrieval-alignment stage over `489,702` query-tool pairs to make hierarchical code generation serve as retrieval.
- The method outperforms BM25, embedding similarity, ToolRetriever, and ToolGen on most `NDCG@k` comparisons.
- The paper treats tool retrieval and tool execution as linked stages, so better retrieval is expected to translate into stronger downstream task completion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
