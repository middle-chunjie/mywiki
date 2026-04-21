---
type: concept
title: Text-Attributed Graph
slug: text-attributed-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [TAG, text-attributed graph, textual graph, 文本属性图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Text-Attributed Graph** (文本属性图) — a graph whose nodes and edges are annotated with natural-language attributes that can be encoded, retrieved, and reasoned over by language models.

## Key Points

- The paper formalizes a textual graph as `` `G = (V, E, {x_n}, {x_e})` ``, where both nodes and edges carry text sequences.
- This representation is the common substrate used for the three benchmark components: commonsense explanation graphs, scene graphs, and Freebase-derived knowledge graphs.
- G-Retriever first embeds node and edge text for retrieval, then textualizes the selected subgraph again before generation.
- The setting is challenging because large text-attributed graphs can exceed the LLM context window if serialized naively.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-gretriever-2402-07630]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-gretriever-2402-07630]].
