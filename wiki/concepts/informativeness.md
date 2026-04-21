---
type: concept
title: Informativeness
slug: informativeness
date: 2026-04-20
updated: 2026-04-20
aliases: [informative response quality, 信息量]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Informativeness** (信息量) — the degree to which a model response provides useful, contentful, and non-trivial information rather than a vague or empty answer.

## Key Points

- The paper evaluates informativeness on TruthfulQA using the benchmark's `%Info` metric in addition to truthfulness.
- EmotionPrompt improves average informativeness by `12%` across ChatGPT, Vicuna-13B, and Flan-T5-Large.
- The authors argue that emotional prompts often elicit more detailed supporting evidence and better-organized answers in open-ended generation.
- Informativeness is treated as complementary to truthfulness: a response can be factually better while also conveying richer task-relevant content.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-large-2307-11760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-large-2307-11760]].
