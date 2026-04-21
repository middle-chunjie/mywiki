---
type: concept
title: Keyword Extraction
slug: keyword-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [query keyword extraction, 关键词抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Keyword Extraction** (关键词抽取) — the process of deriving salient lexical or semantic keys from text or queries so that downstream retrieval or indexing can focus on the most relevant information units.

## Key Points

- LightRAG extracts both local and global keywords from each query before retrieval.
- Local keywords target specific entities and details, while global keywords target higher-level relations or themes.
- These keywords are matched against profiled entity keys and relation keys in the vector database.
- The paper treats keyword extraction as the bridge between natural-language queries and the graph-structured retrieval space.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-lightrag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-lightrag]].
