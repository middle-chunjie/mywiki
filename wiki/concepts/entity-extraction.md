---
type: concept
title: Entity Extraction
slug: entity-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [named entity extraction, entity mining]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Entity Extraction** (实体抽取) — the identification of salient entities in text together with structured attributes such as type, name, and description.

## Key Points

- GraphRAG prompts an LLM to extract important entities from each chunk along with a typed description.
- The extraction prompt can be adapted with domain-specific few-shot examples for specialized corpora.
- The appendix adds iterative self-reflection so the model can recover entities missed in earlier passes.
- Extracted entities become graph nodes and the backbone of the later community summarization pipeline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[edge-2024-local-2404-16130]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[edge-2024-local-2404-16130]].
