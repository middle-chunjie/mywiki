---
type: concept
title: Demonstration Retrieval
slug: demonstration-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [example retrieval, 示例检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Demonstration Retrieval** (示例检索) — the process of selecting informative input-output examples from a labeled pool so they can be inserted into the prompt for in-context learning.

## Key Points

- This paper treats demonstration retrieval as the central bottleneck of in-context learning quality across many NLP tasks.
- UDR learns retrieval from LM feedback rather than relying only on lexical or embedding similarity.
- The paper unifies retrieval supervision from classification, generation, and semantic parsing tasks into one ranking formulation.
- The learned retriever substantially outperforms Random, BM25, SBERT, Instructor, DR-Target, and EPR on broad multitask evaluation.
- The results suggest that demonstration quality matters more than demonstration quantity in many settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-unified]].
