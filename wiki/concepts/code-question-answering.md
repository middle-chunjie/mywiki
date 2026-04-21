---
type: concept
title: Code Question Answering
slug: code-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [code QA, 代码问答]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Question Answering** (代码问答) — deciding whether a candidate code snippet correctly answers a natural-language question about how to perform a programming task.

## Key Points

- CoSQA formulates code question answering as binary classification over a query-code pair with label `1` or `0`.
- The paper evaluates this task on CodeXGLUE WebQueryTest with `1,046` expert-labeled Python query-code pairs.
- Training CodeBERT on CoSQA raises accuracy from `47.80` to `52.87`, and adding CoCLR further improves it to `63.38`.
- Annotation guidelines explicitly distinguish full answers, over-complete answers, partial but sufficient answers, and insufficient answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2021-cosqa-2105-13239]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2021-cosqa-2105-13239]].
