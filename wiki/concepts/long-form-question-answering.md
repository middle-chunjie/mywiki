---
type: concept
title: Long-form Question Answering
slug: long-form-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [LFQA]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-form Question Answering** — a question answering setting where systems produce multi-sentence, explanation-like answers rather than short spans or entities.

## Key Points

- CLAPnq frames LFQA as a grounded task: answers should be supported by a gold passage instead of relying on open-ended world knowledge alone.
- The benchmark is built from Natural Questions items that have long answers but no short answers, making the target responses richer than standard extractive QA.
- Good LFQA answers in this paper must be concise, complete, and cohesive, often combining multiple non-contiguous evidence spans into one fluent response.
- The paper evaluates LFQA under retrieval, gold-passage generation, and full RAG settings rather than only scoring the final answer text.
- CLAPnq also includes unanswerable cases, so LFQA systems are tested on abstention as well as answer composition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rosenthal-2025-clapnq]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rosenthal-2025-clapnq]].
