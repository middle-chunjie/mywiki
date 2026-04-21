---
type: concept
title: Open Information Extraction
slug: open-information-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [OpenIE, open IE, 开放式信息抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Open Information Extraction** (开放式信息抽取) — an information extraction paradigm that converts raw text into open-schema relational tuples or triples without requiring a fixed ontology in advance.

## Key Points

- HippoRAG uses 1-shot LLM prompting to extract a schemaless knowledge graph from each passage rather than relying on a predefined relation schema.
- The pipeline first extracts named entities and then injects them into a second OpenIE prompt to bias extraction toward salient nodes while still allowing general noun-phrase concepts.
- The resulting triples define the graph nodes `N` and relation edges `E` that later support query-time graph search.
- In the paper's ablations, replacing the LLM-based OpenIE stage with REBEL causes large retrieval drops, showing that extraction quality is central to the method.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guti-rrez-2024-hipporag-2405-14831]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guti-rrez-2024-hipporag-2405-14831]].
