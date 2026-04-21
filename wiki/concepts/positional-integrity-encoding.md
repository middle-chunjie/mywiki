---
type: concept
title: Positional Integrity Encoding
slug: positional-integrity-encoding
date: 2026-04-20
updated: 2026-04-20
aliases: [PIE, positional integrity encoding]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Positional Integrity Encoding** — a RoPE-based cache editing method that preserves correct post-edit token positions by algebraically rotating cached suffix keys instead of fully recomputing them.

## Key Points

- PIE targets real-time editing for code LLMs, where a user modifies a local span and the model must continue generation from the edited context.
- The method keeps the unchanged prefix cache, freshly encodes only the edited span, and reuses suffix values while rotating suffix keys by a shared shift matrix.
- Its key update simplifies from `R_(i+m+j'-j) R^(-1)_(j') k^l_{j'}` to `R_(i+m-j) k^l_{j'}`, turning suffix repair into a single matrix transform.
- On RepoBench-C-8k, PIE preserves near-identical accuracy to full recomputation while reducing KV-cache update latency by more than `85%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-let-2407-03157]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-let-2407-03157]].
