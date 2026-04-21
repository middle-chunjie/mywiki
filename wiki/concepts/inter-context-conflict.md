---
type: concept
title: Inter-Context Conflict
slug: inter-context-conflict
date: 2026-04-20
updated: 2026-04-20
aliases: [context-context conflict, 上下文间冲突]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Inter-Context Conflict** (上下文间冲突) — a knowledge conflict where multiple contextual sources within the prompt or retrieved evidence disagree with one another.

## Key Points

- The survey treats this as a central failure mode of [[retrieval-augmented-generation]] systems that combine multiple retrieved documents.
- It highlights `2` common causes: misinformation among retrieved documents and clashes between outdated and updated evidence.
- Surveyed LLMs are biased toward query-relevant, memory-aligned, frequent, and early-positioned evidence when resolving contradictory contexts.
- Reviewed solutions include contradiction-detection models, tool-augmented fact-checking, discriminator-based robustness training, and query augmentation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-knowledge-2403-08319]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-knowledge-2403-08319]].
