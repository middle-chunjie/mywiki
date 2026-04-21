---
type: concept
title: Human-Labeled Docstring
slug: human-labeled-docstring
date: 2026-04-20
updated: 2026-04-20
aliases: [manually written docstring, 人工标注文档字符串]
tags: [prompting, benchmark, code-generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Human-labeled docstring** (人工标注文档字符串) — a manually authored natural-language function description used as an alternative prompt to the original in-repository docstring.

## Key Points

- CoderEval adds a human-labeled docstring to each task to reduce leakage from potentially memorized original comments.
- The paper reports that `13` experienced engineers wrote these docstrings, with additional checking to improve quality.
- The benchmark uses the paired prompts to study how prompt wording changes code-generation effectiveness under roughly fixed semantics.
- Prompt choice materially affects results, especially in Java, where original versus human-labeled docstrings lead to noticeably different `Pass@k` values.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-codereval-2302-00288]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-codereval-2302-00288]].
