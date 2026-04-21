---
type: concept
title: Syntactic Filter
slug: syntactic-filter
date: 2026-04-20
updated: 2026-04-20
aliases: [rule-based syntactic filter, syntax filter, 句法过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Syntactic Filter** (句法过滤) — a rule-driven filter that accepts or rejects text based on explicit surface-form patterns rather than learned semantic judgments.

## Key Points

- The paper’s first-stage filter removes obvious non-query artifacts before semantic modeling.
- Eight rules cover HTML tags, parentheses, Javadoc tags, URLs, non-English content, punctuation-only strings, interrogations, and short sentences.
- Some matched features are stripped in place while others trigger full rejection of the comment-code pair.
- The rules are designed to be conservative, non-overlapping, and extensible.
- Ablation shows that removing this filter causes measurable drops on both DeepCS and CARLCS.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-importance-2202-06649]].
