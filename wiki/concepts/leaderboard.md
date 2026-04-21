---
type: concept
title: Leaderboard
slug: leaderboard
date: 2026-04-20
updated: 2026-04-20
aliases: [排行榜]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Leaderboard** (排行榜) — a benchmark interface that ranks systems on a shared task using one or more evaluation metrics.

## Key Points

- The paper argues that a leaderboard should do more than report a single aggregate score; it should also support comparison of models and inspection of informative examples.
- DAD reinterprets a leaderboard as a joint measurement problem over systems and evaluation items rather than a flat table of averages.
- The authors treat declining informativeness as a signal that a benchmark may have outlived its usefulness and should be refreshed or retired.
- The proposed leaderboard design is meant to guide annotation, surface flawed items, and make benchmarking more scientifically useful.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
