---
type: concept
title: Time Awareness
slug: time-awareness
date: 2026-04-20
updated: 2026-04-20
aliases: [time-aware, time awareness, 时间感知]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Time Awareness** (时间感知) — the capability to recognize whether an input depends on facts whose correct answers change over time.

## Key Points

- UAR treats time awareness as an independent retrieval criterion instead of folding it into generic uncertainty or factuality.
- The time-aware classifier is trained with TAQA questions as positives and TriviaQA questions as negatives.
- In UAR-Criteria, time-sensitive questions trigger retrieval even if the model might otherwise appear confident.
- This criterion is critical for the strong TAQA and FreshQA results reported in the paper.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-unified-2406-12534]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-unified-2406-12534]].
