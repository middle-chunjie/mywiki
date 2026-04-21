---
type: concept
title: Blank Filling
slug: blank-filling
date: 2026-04-20
updated: 2026-04-20
aliases: [fill-in-the-blank evaluation, template filling, 填空评测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Blank Filling** (填空评测) — an evaluation format in which a model must complete predefined slots in a template, and the filled content is scored against structured criteria.

## Key Points

- InfiBench uses blank filling when open-ended answers are too unconstrained to grade reliably.
- The filled blanks may correspond to either natural-language spans or code snippets.
- Each blank can be checked with keywords, regular expressions, or recursive logical rules defined by domain experts.
- Blank filling accounts for `12.22%` of benchmark questions in the final release.
- The evaluation toolkit extracts filled content with longest common subsequence matching, making the metric fully automatic.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-infibench-2404-07940]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-infibench-2404-07940]].
