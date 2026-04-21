---
type: concept
title: Global-to-Local Modeling
slug: global-to-local-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [global-to-local language modeling, 全局到局部建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Global-to-Local Modeling** (全局到局部建模) — a modeling strategy that first captures broad sequence context at a coarse granularity and then resolves fine-grained predictions using local computation conditioned on that coarse context.

## Key Points

- Block Transformer compresses previous blocks into a context embedding and lets the token decoder use that embedding as its only global signal.
- The design separates global comprehension from local token interaction, so upper layers never need to attend over the full prompt.
- For `L = 2048` and `L_B = 4`, the local decoder keeps only the current block in KV cache while the block decoder retains coarse global state.
- The paper shows this global-to-local design can preserve useful long-context behavior on PG19 and Needle-In-a-Haystack despite aggressive locality in upper layers.
- Throughput rises rapidly as block length increases, while language-model loss degrades more gradually, producing a favorable throughput-quality frontier.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ho-2024-block-2406-02657]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ho-2024-block-2406-02657]].
