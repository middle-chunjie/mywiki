---
type: concept
title: Hierarchical Monitoring
slug: hierarchical-monitoring
date: 2026-04-20
updated: 2026-04-20
aliases: [hierarchical summarization for monitoring, 分层监控]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Monitoring** (分层监控) — a monitoring architecture that recursively summarizes and scores local trajectory chunks before aggregating them into a global suspiciousness judgment.

## Key Points

- The method treats long agent trajectories as a hierarchy of smaller interactions.
- It is designed to solve needle-in-the-haystack detection in trajectories averaging over `20,000` tokens.
- The paper derives a computational reduction from `L^2` attention to `(L / N) x N^2`.
- Hierarchical monitoring is more resistant than the baseline to some prompt-injection attacks because manipulative text can be diluted or isolated within chunks.
- Its standalone behavior is dataset-sensitive, which motivates combining it with sequential monitoring.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
