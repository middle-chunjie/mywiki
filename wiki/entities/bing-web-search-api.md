---
type: entity
title: Bing Web Search API
slug: bing-web-search-api
date: 2026-04-20
entity_type: tool
aliases: [Bing Search API, Bing Web Search]
tags: []
---

## Description

Bing Web Search API is the web retrieval backend used by Search-o1. The paper uses it to fetch the top-10 US-EN web results whenever the reasoning model emits a search query.

## Key Contributions

- Supplies external web documents for each agentic retrieval step.
- Lets Search-o1 retrieve evidence multiple times inside a single reasoning trace.

## Related Concepts

- [[web-search]]
- [[active-retrieval]]
- [[retrieval-augmented-generation]]

## Sources

- [[li-2025-searcho-2501-05366]]
