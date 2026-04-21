---
type: concept
title: Contrastive Search
slug: contrastive-search
date: 2026-04-20
updated: 2026-04-20
aliases: [CS, 对比搜索]
tags: [decoding, generation, search]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Contrastive Search** (对比搜索) — a decoding strategy for neural text generation that balances model confidence with degeneration avoidance by contrasting likely continuations against repetitive ones.

## Key Points

- The paper includes contrastive search as a deterministic alternative to random sampling for small-model question generation.
- The authors expected CS to be competitive with beam search, but its empirical `hitsR` is lower in their benchmark.
- CS is also substantially slower than random sampling in the reported hits-per-second comparison.
- The paper leaves open whether better CS hyperparameters could narrow the observed quality gap.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
