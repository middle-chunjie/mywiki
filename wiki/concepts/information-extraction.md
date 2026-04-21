---
type: concept
title: Information Extraction
slug: information-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [IE, information extraction, 信息抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information Extraction** (信息抽取) — the task of converting unstructured text into structured outputs such as entities, relations, events, and arguments.

## Key Points

- [[ma-2023-large-2303-08559]] treats information extraction as a broad family spanning NER, RE, ED, and EAE rather than a single narrow task.
- The paper argues that direct LLM prompting is generally not competitive for few-shot IE once labeled data is moderately available.
- Fine-grained IE schemas are especially challenging for LLMs because prompt instructions and demonstrations do not scale well with many labels.
- The paper proposes combining supervised SLM filters with LLM rerankers instead of relying on LLMs as standalone IE systems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2023-large-2303-08559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2023-large-2303-08559]].
