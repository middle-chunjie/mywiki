---
type: concept
title: Trajectory Summarization
slug: trajectory-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [trajectory compression, search-trajectory summarization, 轨迹摘要]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Trajectory Summarization** (轨迹摘要) — the compression of an agent's multi-step interaction history into a shorter summary that preserves useful state for future steps.

## Key Points

- [[yen-2025-lost-2510-18939]] summarizes the entire conversation history rather than summarizing only individual search results.
- Slim applies summarization every `n = 50` tool turns by default and replaces the current context with the resulting summary.
- The method is intended to support longer search horizons without saturating the model's context window.
- Ablations compare fixed-period summarization with more aggressive `n = 25` and token-threshold triggers at `32,768` and `65,536` tokens.
- The paper treats trajectory summarization as a simple but effective context-management primitive for long-horizon agents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2025-lost-2510-18939]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2025-lost-2510-18939]].
