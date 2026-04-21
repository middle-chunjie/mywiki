---
type: concept
title: Asynchronous Tool Calling
slug: asynchronous-tool-calling
date: 2026-04-20
updated: 2026-04-20
aliases: [async tool use, 异步工具调用]
tags: [agents, systems, tool-use]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Asynchronous Tool Calling** (异步工具调用) — a tool-execution strategy where model generation and external API requests are overlapped so rollouts do not synchronously block on tool latency.

## Key Points

- DR Tulu sends a tool request immediately when a rollout emits one, instead of waiting for the whole batch to finish generating.
- While the tool request is in flight, the corresponding rollout is put to sleep and compute is reallocated to other generations.
- This overlap reduces end-to-end RL wall-clock time in a setting with many web and paper search calls.
- The paper implements the mechanism inside `dr-agent-lib`, which also manages concurrency limits and caching.
- Even with this optimization, RL training remains partly bottlenecked by external API rate limits, showing the systems importance of async execution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2025-dr-2511-19399]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2025-dr-2511-19399]].
