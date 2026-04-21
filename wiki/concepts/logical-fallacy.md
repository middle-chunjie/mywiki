---
type: concept
title: Logical Fallacy
slug: logical-fallacy
date: 2026-04-20
updated: 2026-04-20
aliases: [logic fallacy, 逻辑谬误]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Logical Fallacy** (逻辑谬误) — an inconsistency where the logical relation expressed in an answer conflicts with, or is unsupported by, the logical structure of the source evidence.

## Key Points

- Face4RAG treats logical fallacy as a distinct class of factual inconsistency rather than a side effect of unsupported facts.
- The paper defines four main logical fallacy types: overgeneralization, causal confusion, confusing sufficient and necessary conditions, and inclusion relation error.
- Existing FCE pipelines are criticized for breaking logical dependencies during answer decomposition and therefore missing these errors.
- L-Face4RAG adds an explicit logic-consistency stage and shows the largest accuracy gains on logical-fallacy categories such as `LCaus.` and `LOth.`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-facerag-2407-01080]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-facerag-2407-01080]].
