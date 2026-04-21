---
type: concept
title: Entity-Relationship Extraction
slug: entity-relationship-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [entity relation extraction, 实体关系抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Entity-Relationship Extraction** (实体关系抽取) — the process of identifying entities and the semantic relations connecting them from unstructured text in order to build structured representations.

## Key Points

- LightRAG uses an LLM to extract nodes and edges from each text chunk before graph construction.
- The extracted entities and relations become the basis of the graph index used for downstream retrieval.
- Chunking is essential because extraction is run independently over smaller document segments rather than over full documents.
- The paper treats extraction quality as foundational, since later profiling, deduplication, and retrieval all depend on the extracted graph elements.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-lightrag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-lightrag]].
