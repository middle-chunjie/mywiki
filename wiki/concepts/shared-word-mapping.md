---
type: concept
title: Shared Word Mapping
slug: shared-word-mapping
date: 2026-04-20
updated: 2026-04-20
aliases: [shared vocabulary mapping, SWM, 共享词映射]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Shared Word Mapping** (共享词映射) — using one common vocabulary and embedding table for multiple text-like inputs so identical words receive consistent vector representations across modalities.

## Key Points

- TranCS applies a single vocabulary to both translated code descriptions and natural-language comments.
- The embedding matrix has vocabulary size `15,000` and word dimension `512`.
- The paper motivates SWM as a fix for embedding divergence caused by separate vocabularies for comments and code features.
- Full TranCS with SWM substantially outperforms the version with context-aware translation alone.
- The same principle also gives smaller gains when applied to token-only or DeepCS-style models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-code-2202-08029]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-code-2202-08029]].
