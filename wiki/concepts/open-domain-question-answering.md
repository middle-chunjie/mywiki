---
type: concept
title: Open-Domain Question Answering
slug: open-domain-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [open-domain QA]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Open-Domain Question Answering** (开放域问答) — a question-answering setting in which systems answer arbitrary questions by relying on broad external knowledge rather than a fixed provided context.

## Key Points

- SELF-RAG is evaluated on PopQA and TriviaQA as open-domain QA tasks.
- The method uses retrieval-on-demand to decide whether a question should be answered from retrieved passages or from the model's internal knowledge.
- Analysis in the paper shows SELF-RAG's correct answers are far more often explicitly present in retrieved evidence than those of baseline LMs.
- The framework is motivated partly by the mismatch between open-domain QA, which benefits from retrieval, and open-ended instructions, which may not.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[asai-2023-selfrag-2310-11511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[asai-2023-selfrag-2310-11511]].
