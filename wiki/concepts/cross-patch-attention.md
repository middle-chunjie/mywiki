---
type: concept
title: Cross-Patch Attention
slug: cross-patch-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [cross patch local attention, 跨 patch 注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Patch Attention** (跨 patch 注意力) — an attention mechanism that allows a local sequence model to access selected states from neighboring patches instead of restricting attention strictly within the current patch.

## Key Points

- In MEGABYTE, the local model can concatenate the previous patch's last `r` keys and values into each self-attention layer to expand short-range context cheaply.
- The extension uses rotary embeddings to represent relative positions across the patch boundary.
- The paper frames this as a way to give the local model more immediate context while letting the global model focus on longer-range structure.
- Ablations show the base architecture remains fairly robust without this component, so cross-patch attention is helpful but not strictly required for the method to work.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-megabyte-2305-07185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-megabyte-2305-07185]].
