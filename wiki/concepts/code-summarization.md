---
type: concept
title: Code Summarization
slug: code-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [source code summarization]
tags: [software-engineering, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Summarization** (代码摘要) — the task of generating a natural-language description that captures the functionality of a source-code snippet.

## Key Points

- [[ahmad-2021-unified]] fine-tunes PLBART to summarize Ruby, JavaScript, Go, Python, Java, and PHP code into English.
- The paper evaluates this task with smoothed BLEU-4 and reports an average score of `18.32` for PLBART.
- Gains are strongest on Ruby, the smallest training set, which the paper interprets as evidence of better transfer from pretraining.
- PHP is the main weak point, suggesting that mismatch between pretraining languages and target-language syntax matters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ahmad-2021-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ahmad-2021-unified]].
