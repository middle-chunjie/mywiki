---
type: concept
title: Self-Supported Question Answering
slug: self-supported-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [SQA, self supported question answering, 自支持问答]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Supported Question Answering** (自支持问答) — a question answering setup in which the model must produce an answer together with explicit evidence sufficient for a human to judge whether the answer is supported.

## Key Points

- [[menick-2022-teaching-2203-11147]] formalizes SQA as generating both free-form claims and inline evidence rather than an answer alone.
- The paper evaluates SQA with two human judgments: whether the response is plausible and whether the provided quote evidence supports the full answer.
- GopherCite treats SQA as conditional language modeling, enabling the answer and evidence to be produced in one sequence.
- The paper argues SQA is a reusable subtask for broader systems such as dialogue, debate, and open-book assistance.
- Results on NaturalQuestionsFiltered and ELI5Filtered show that SQA quality improves substantially with reward-model reranking and RL from human preferences.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[menick-2022-teaching-2203-11147]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[menick-2022-teaching-2203-11147]].
