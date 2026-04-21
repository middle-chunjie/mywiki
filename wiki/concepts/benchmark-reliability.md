---
type: concept
title: Benchmark Reliability
slug: benchmark-reliability
date: 2026-04-20
updated: 2026-04-20
aliases: [evaluation reliability, 基准可靠性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Benchmark Reliability** (基准可靠性) — the extent to which an evaluation benchmark yields stable system rankings across samples, replications, or closely related test settings.

## Key Points

- The paper evaluates reliability by comparing rankings from small sampled test sets against rankings from larger or different splits using Kendall correlation.
- DAD improves reliability because it weights items by inferred informativeness rather than treating all examples as equally valuable.
- The advantage is strongest at small sample sizes, where naive average accuracy is especially noisy.
- The work frames reliability as a first-class property of leaderboards, not merely a post hoc statistical diagnostic.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
