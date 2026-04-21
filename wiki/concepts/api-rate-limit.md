---
type: concept
title: API Rate Limit
slug: api-rate-limit
date: 2026-04-20
updated: 2026-04-20
aliases: [API rate limiting, API 速率限制]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**API Rate Limit** (API 速率限制) — a service-side cap on request frequency that constrains throughput and can dominate end-to-end latency in hosted model inference.

## Key Points

- The paper treats API request overhead and requests-per-minute limits as major practical bottlenecks for LLM deployments.
- Batch prompting reduces the number of requests from `N` to roughly `N / b`, directly lowering exposure to rate limits.
- The measured wall-clock savings include both model generation time and service-side waiting or overhead.
- The authors caution that if backend decoding becomes the main latency source, rate-limit savings alone will no longer guarantee large time improvements.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2023-batch-2301-08721]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2023-batch-2301-08721]].
