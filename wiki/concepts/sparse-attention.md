---
type: concept
title: Sparse Attention
slug: sparse-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [sparse attention, 稀疏注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sparse Attention** (稀疏注意力) — an attention pattern that restricts which token pairs can interact, reducing full all-to-all attention while preserving selected dependencies.

## Key Points

- [[ratner-2023-parallel-2212-10947]] uses a structured sparse mask in which context tokens attend only within their own window replica.
- The same paper keeps task tokens globally connected to all windows, yielding a hybrid pattern: sparse among context windows, dense from task tokens backward.
- This masking is the mechanism that prevents collisions among reused position IDs across windows.
- In PCW, sparsity lets accessible context grow linearly with the number of windows `B` rather than requiring a monolithic longer dense context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ratner-2023-parallel-2212-10947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ratner-2023-parallel-2212-10947]].
