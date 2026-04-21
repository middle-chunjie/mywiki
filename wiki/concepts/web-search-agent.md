---
type: concept
title: Web-Search Agent
slug: web-search-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [web search agent, search agent, 网页搜索代理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Web-Search Agent** (网页搜索代理) — an external search module that reformulates a reasoning query, retrieves web pages, reranks them, and synthesizes grounded evidence for reinsertion into the main reasoning process.

## Key Points

- The agent uses query breakdown to convert a broad research request into one or more search-engine-friendly queries.
- Search results are reranked against the original query and reasoning context before RAG synthesis.
- The paper uses a top-10 average relevance threshold to decide whether another round of query refinement is needed.
- The final returned snippet is conditioned on both the original query and the Mind-Map-derived context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-agentic-2502-04644]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-agentic-2502-04644]].
