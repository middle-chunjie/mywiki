---
type: concept
title: Retrieval Question Answering
slug: retrieval-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [ReQA, retrieval QA]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Question Answering** (检索式问答) — a question answering setup in which a system retrieves supporting documents from a corpus and then uses them to generate or extract an answer.

## Key Points

- [[yang-2023-prca]] formulates the task as a retriever-generator pipeline where the input evidence is the `Top-K` documents returned for a query.
- The paper emphasizes that ReQA quality depends not only on retrieval recall but also on how well the generator can use long, noisy retrieved context.
- PRCA targets ReQA specifically by distilling retrieved evidence before it reaches a frozen black-box generator.
- The evaluation spans three ReQA regimes: single-hop (SQuAD), multi-hop (HotpotQA), and conversational topic-switching QA (TopiOCQA).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-prca]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-prca]].
