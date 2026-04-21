---
type: concept
title: Batch Prompting
slug: batch-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [batch prompting, 批量提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Batch Prompting** (批量提示) — a prompting strategy that groups multiple inference samples into one indexed prompt so a large language model can answer them in a single call.

## Key Points

- The paper formats one batch as ordered question slots `Q[1] ... Q[b]` and aligned answer slots `A[1] ... A[b]`.
- Shared few-shot exemplars are also grouped into batches, so the same demonstration tokens are amortized across multiple test instances.
- With `N` evaluation samples and batch size `b`, the number of API calls drops from `N` to roughly `N / b`.
- The reported token and latency savings reach up to `5x` at `b = 6`, though larger batches often reduce accuracy.
- The method works not only with Chain-of-Thought prompting but also with end-to-end and program-based reasoning setups.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2023-batch-2301-08721]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2023-batch-2301-08721]].
