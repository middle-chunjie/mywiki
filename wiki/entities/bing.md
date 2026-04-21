---
type: entity
title: Bing
slug: bing
date: 2026-04-20
entity_type: tool
aliases: [Microsoft Bing]
tags: []
---

## Description

Bing is the web search engine used by the Web-Search agent in [[wu-2025-agentic-2502-04644]]. The system retrieves the top-ranked pages from Bing before reranking and RAG-based synthesis.

## Key Contributions

- Supplies the initial top `20` web pages for each refined search query.
- Serves as the retrieval backend for the paper's search-augmented reasoning pipeline.

## Related Concepts

- [[web-search-agent]]
- [[query-breakdown]]
- [[reranking]]

## Sources

- [[wu-2025-agentic-2502-04644]]
