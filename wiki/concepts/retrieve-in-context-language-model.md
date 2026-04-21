---
type: concept
title: Retrieve-in-Context Language Model
slug: retrieve-in-context-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [RIC-LM, retrieve in context LM, 检索入上下文语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieve-in-Context Language Model** (检索入上下文语言模型) — a retrieval-based language model that injects retrieved documents directly into the prompt context of an otherwise standard LM at inference time.

## Key Points

- The paper focuses on RIC-LMs because they can use off-the-shelf retrievers and black-box or minimally modified LMs.
- Retrieved passages are chunked to `256` words and the top `k = 3` documents are prepended before the few-shot examples and query.
- This formulation lets the study isolate datastore scale without changing the LM architecture, unlike RETRO-style specialized models.
- RIC-LMs benefit strongly on knowledge-intensive QA, where a smaller model with retrieval can beat a larger LM-only baseline.
- The paper argues that trillion-token-scale datastore results are especially relevant for this simple and widely used retrieval-in-context paradigm.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2024-scaling-2407-12854]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2024-scaling-2407-12854]].
