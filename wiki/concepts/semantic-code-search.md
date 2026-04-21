---
type: concept
title: Semantic Code Search
slug: semantic-code-search
date: 2026-04-20
updated: 2026-04-20
aliases: [code search, semantic code retrieval, 语义代码搜索]
tags: [code, retrieval, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semantic Code Search** (语义代码搜索) — the task of retrieving code snippets from natural-language descriptions of functionality rather than exact lexical matches.

## Key Points

- This paper frames semantic code search as a compositional reasoning problem over entities and actions, not just embedding similarity between query and code.
- NS3 uses a semantic parse of the query to decompose retrieval into smaller subproblems handled by specialized neural modules.
- The model is evaluated in a reranking setup over CodeBERT candidates on CodeSearchNet and CoSQA/WebQueryTest.
- Reported gains are strongest when training data is limited, suggesting structured decomposition can improve sample efficiency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[arakelyan-2022-ns-2205-10674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[arakelyan-2022-ns-2205-10674]].
