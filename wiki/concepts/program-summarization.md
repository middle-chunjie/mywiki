---
type: concept
title: Program Summarization
slug: program-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [code summarization, 程序摘要]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Program Summarization** (程序摘要) — the task of generating a concise natural-language or tokenized summary of code, often by predicting a function name or description from the program body.

## Key Points

- This paper instantiates adversarial code attacks on the benchmark where a model predicts a function's name from its body.
- The function name is removed from the input, so the model must infer intent from code tokens alone.
- The experiments use roughly `150K` Python functions and `700K` Java functions that are preprocessed into per-function examples.
- The paper evaluates adversarial robustness by measuring how obfuscation changes the predicted summary while leaving code functionality unchanged.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
