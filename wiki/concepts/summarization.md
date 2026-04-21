---
type: concept
title: Summarization
slug: summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [text summarization, 摘要生成]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Summarization** (摘要生成) — the task of generating a shorter text that preserves the main information of a longer source document or dialogue.

## Key Points

- [[li-2023-compressing]] uses summarization as one of the main downstream tasks for evaluating whether compressed context still supports global understanding.
- Among the four tasks in the paper, summarization degrades more gradually than reconstruction as compression increases.
- Human evaluation over `1150` summaries suggests instruct-tuned Vicuna handles compressed context better than base LLaMA.
- The case study shows compressed context can change which details appear in a summary even when the resulting text remains factually acceptable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-compressing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-compressing]].
