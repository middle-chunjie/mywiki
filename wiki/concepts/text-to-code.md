---
type: concept
title: Text-to-Code
slug: text-to-code
date: 2026-04-20
updated: 2026-04-20
aliases: [text to code, text-to-code generation, 文本到代码生成]
tags: [llm, code-generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Text-to-Code** (文本到代码生成) - the task of generating an executable or checkable program from a natural-language specification.

## Key Points

- The paper frames its core problem as generating a target-language program `t in T` from a task description.
- It studies a hard setting where the target language is a VLPL with little representation in LLM pretraining.
- SPEAC decomposes text-to-code into generation in a familiar parent language, static repair, and compilation into the target language.
- Evaluation uses `33` textbook benchmarks and treats parse rate as the main measure of syntactic success.
- Semantic usefulness is judged separately on a `1-5` manual scoring scale for outputs that compile.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mora-2024-synthetic-2406-03636]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mora-2024-synthetic-2406-03636]].
