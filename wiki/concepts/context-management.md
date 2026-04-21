---
type: concept
title: Context Management
slug: context-management
date: 2026-04-20
updated: 2026-04-20
aliases: [context control, 上下文管理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Management** (上下文管理) — the set of strategies used to control what information remains in a model's working context across multi-step inference.

## Key Points

- [[yen-2025-lost-2510-18939]] argues that poor context management is the primary bottleneck in long-horizon agentic search.
- The paper attributes failures such as context overflow, wasted tool calls, and premature stopping to noisy information accumulation.
- Slim improves context management by separating lightweight search snippets from targeted browsing.
- It further compresses the full trajectory every `50` turns instead of only summarizing per-search outputs.
- Better context management is linked to fewer hallucinations and more efficient use of long search trajectories.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2025-lost-2510-18939]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2025-lost-2510-18939]].
