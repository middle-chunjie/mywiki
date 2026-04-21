---
type: concept
title: Meta Knowledge
slug: meta-knowledge
date: 2026-04-20
updated: 2026-04-20
aliases: [metadata for retrieval, 元知识]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Meta Knowledge** (元知识) — auxiliary information about retrieved evidence, rather than the evidence content itself, used to guide downstream generation or decision making.

## Key Points

- MK-TOD defines meta knowledge over retrieved entities using three signals: retrieval order, retrieval confidence, and dialogue-history co-occurrence.
- The paper injects meta knowledge through discrete prefix tokens, natural-language prompts, or a contrastive objective that rewards entity-aware generation.
- Meta knowledge is intended to help the generator discriminate among highly similar KB entries whose attribute values differ only slightly.
- Analysis in the paper suggests meta knowledge makes generator quality align more closely with retriever quality than in standard retrieve-then-generate TOD systems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shen-2023-retrievalgeneration]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shen-2023-retrievalgeneration]].
