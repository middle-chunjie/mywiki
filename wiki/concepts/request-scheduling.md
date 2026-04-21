---
type: concept
title: Request Scheduling
slug: request-scheduling
date: 2026-04-20
updated: 2026-04-20
aliases: [scheduling, 请求调度]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Request Scheduling** (请求调度) — the policy layer that decides how inference requests are batched, ordered, preempted, and mapped to compute resources during online model serving.

## Key Points

- The survey argues that LLM serving needs specialized scheduling because autoregressive generation has iterative execution, unknown output lengths, and stateful context management.
- It highlights Orca as an early shift from request-level scheduling to iteration-level scheduling with selective batching.
- The paper groups later systems around continuous or inflight batching, chunked prefill, preemption, and disaggregated prefill/decode execution.
- Mixed workloads make scheduling harder, so several systems add length prediction, fairness objectives, priority handling, or load balancing to improve throughput and job completion time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[miao-2023-efficient-2312-15234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[miao-2023-efficient-2312-15234]].
