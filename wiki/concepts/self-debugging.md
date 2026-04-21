---
type: concept
title: Self-Debugging
slug: self-debugging
date: 2026-04-20
updated: 2026-04-20
aliases: [self debug, 自调试]
tags: [llm, code, debugging]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Debugging** (自调试) — a prompting strategy in which a model inspects its own predicted code, receives automatic feedback, and iteratively revises the program without extra training.

## Key Points

- [[chen-2023-teaching-2304-05128]] structures self-debugging into Generation, Explanation, and Feedback steps.
- The framework reuses an initial failed prediction instead of discarding it and asking for completely fresh candidates.
- Feedback can be simple correctness labels, unit-test outputs, code explanations, or execution traces.
- The method is training-free and relies only on prompting plus optional code execution.
- The paper shows substantial gains on Spider, TransCoder, and MBPP, especially when execution-derived feedback is available.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-teaching-2304-05128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-teaching-2304-05128]].
