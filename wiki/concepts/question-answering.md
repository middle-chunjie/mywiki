---
type: concept
title: Question Answering
slug: question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [QA, 问答]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Question Answering** (问答) — the task of predicting an answer to a question from given context, knowledge, or external evidence.

## Key Points

- [[gou-2023-diversify]] uses a generative QA model as an external evaluator for question consistency.
- The consistency reward is derived from QA loss on the gold answer, making answerability a direct training signal for question generation.
- QA-based EM and F1 are also used in ablation analysis to study the diversity-consistency trade-off.
- The paper relies on the duality between QG and QA: a good generated question should make the correct answer recoverable from the source context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2023-diversify]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2023-diversify]].
