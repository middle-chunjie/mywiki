---
type: concept
title: Dense Passage Retrieval
slug: dense-passage-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [DPR, 稠密段落检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dense Passage Retrieval** (稠密段落检索) — a retrieval approach that embeds queries and candidates into dense vectors and ranks candidates by similarity in embedding space.

## Key Points

- [[gou-2023-diversify]] uses DPR as the template retriever for question style templates rather than for document retrieval.
- Two BERT encoders separately embed candidate templates and query templates, with similarity computed by inner product.
- The retriever is initialized from pretrained DPR weights and then updated jointly with the generator under RL rewards.
- The paper argues this end-to-end optimization makes retrieval favor templates that improve both diversity and consistency, not just lexical similarity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2023-diversify]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2023-diversify]].
