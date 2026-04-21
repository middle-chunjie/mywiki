---
type: concept
title: Real-Time Evaluation
slug: real-time-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [online evaluation, live evaluation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Real-Time Evaluation** (实时评测) — an evaluation setup in which systems are tested within a live time window using information that becomes available during deployment rather than a retrospectively frozen corpus.

## Key Points

- REALTIME QA opens a submission window immediately after weekly question release and closes it when the next batch appears.
- The benchmark supports retroactive comparison only if a system is restricted to evidence that would have been available during the original evaluation window.
- The paper shows that measured performance depends on submission time because retrieval freshness changes after question release.
- This setup operationalizes temporal validity as part of the metric, not merely as dataset metadata.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kasai-2024-realtime-2207-13332]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kasai-2024-realtime-2207-13332]].
