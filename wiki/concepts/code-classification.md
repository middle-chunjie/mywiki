---
type: concept
title: Code Classification
slug: code-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [source code classification, 代码分类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Classification** (代码分类) — assigning discrete labels to source-code snippets, such as bug status, algorithmic complexity, or runtime error type, from learned code representations.

## Key Points

- The paper evaluates encoder transfer by freezing CodeSage and training only a linear head for downstream classification.
- Benchmarks cover defect detection in C, algorithmic complexity prediction in Java, and runtime error prediction in Python.
- CodeSage substantially improves over prior encoders on complexity and runtime prediction, especially as model size increases.
- Frozen CodeSage underperforms OpenAI Ada-002 on defect detection, showing that stronger retrieval embeddings do not automatically dominate every classification task.
- Full-model finetuning closes that gap and lifts CodeSage-large to macro-F1 `66.38`, `96.20`, and `49.25` on the three tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code-2402-01935]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code-2402-01935]].
