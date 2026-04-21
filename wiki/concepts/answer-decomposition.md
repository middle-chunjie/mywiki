---
type: concept
title: Answer Decomposition
slug: answer-decomposition
date: 2026-04-20
updated: 2026-04-20
aliases: [response decomposition, 答案分解]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Answer Decomposition** (答案分解) — the process of splitting a long generated answer into smaller evaluation units so that each unit can be checked more reliably against evidence.

## Key Points

- Face4RAG argues that naive decomposition can erase semantic and logical dependencies that are necessary for detecting logical fallacies.
- L-Face4RAG therefore decomposes only when spans are not tightly linked semantically or logically.
- The method explicitly replaces pronouns with referents so each segment can be judged independently against the reference.
- Preserving original sentence structure is treated as a design principle to reduce extra hallucination introduced by decomposition itself.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-facerag-2407-01080]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-facerag-2407-01080]].
