---
type: concept
title: Sparse Transformer
slug: sparse-transformer
date: 2026-04-20
updated: 2026-04-20
aliases: [sparse attention transformer, 稀疏 Transformer]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sparse Transformer** (稀疏 Transformer) — a Transformer variant that restricts the attention pattern to a structured subset of token pairs in order to reduce the time and memory cost of long-sequence modeling.

## Key Points

- LongCoder replaces dense causal attention with a combination of window, bridge, and global attention masks.
- The design keeps the dominant complexity near linear in sequence length instead of quadratic.
- Unlike generic sparse attention baselines, LongCoder injects code-specific inductive bias through syntax-selected global positions.
- The paper shows that sparse attention improves both long-context completion accuracy and efficiency relative to dense `512`-token baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2023-longcoder-2306-14893]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2023-longcoder-2306-14893]].
