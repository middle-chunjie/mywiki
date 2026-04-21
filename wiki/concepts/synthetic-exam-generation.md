---
type: concept
title: Synthetic Exam Generation
slug: synthetic-exam-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [task-specific exam generation, synthetic exam construction, 合成考试生成]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Synthetic Exam Generation** (合成考试生成) — the automatic construction of task-specific exam questions from source documents so a model can be evaluated against a corpus without manually labeled test data.

## Key Points

- The paper generates multiple-choice questions from each task corpus document rather than collecting human-written evaluation sets.
- Question generation is document-grounded, making the resulting exam specific to the target corpus instead of a generic benchmark.
- The generator is followed by parsing, self-containment, and discriminator-quality filters because raw LLM outputs are not reliable enough to use directly.
- The final exams retain `275`, `381`, `148`, and `515` questions on the four benchmark tasks after filtering and pruning.
- The same exam can be reused both for predictive ranking of pipelines and for prescriptive analysis of which RAG design choices matter most.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guinet-2024-automated-2405-13622]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guinet-2024-automated-2405-13622]].
