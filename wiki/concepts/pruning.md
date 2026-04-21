---
type: concept
title: Pruning
slug: pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [model pruning, structured pruning, 剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pruning** (剪枝) — a model compression procedure that removes layers, heads, hidden dimensions, or intermediate widths while trying to preserve downstream behavior.

## Key Points

- DRAMA prunes Llama-3.2-1B into `0.1B` and `0.3B` decoder-only backbones instead of training small encoder backbones from scratch.
- The paper uses structured masks over layers, hidden units, attention heads, and intermediate dimensions, optimized with hard-concrete variables and constraint penalties.
- Pruning is performed in two stages: a constrained pruning stage and a recovery stage with continued language-model pretraining.
- The reported cost is about `0.5B` tokens for pruning and `26B` tokens for recovery pretraining across English plus `19` non-English languages.
- The pruned models retain multilingual and long-context priors, which the paper argues explains their advantage over similarly sized encoder-only retrievers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ma-2025-drama-2502-18460]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ma-2025-drama-2502-18460]].
