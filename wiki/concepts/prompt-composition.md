---
type: concept
title: Prompt Composition
slug: prompt-composition
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt composition, 提示组合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Composition** (提示组合) - the procedure for assembling multiple context fragments into a single prompt under a fixed token budget.

## Key Points

- RLPG prepends the selected proposal context before the default Codex left context.
- The default allocation gives half of the prompt budget to proposal context and half to the default context.
- For post-lines proposals, the paper also studies `1/4` and `3/4` allocations to the proposal side.
- When either side exceeds its budget, the context is truncated according to source-specific heuristics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shrivastava-2023-repositorylevel-2206-12839]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shrivastava-2023-repositorylevel-2206-12839]].
