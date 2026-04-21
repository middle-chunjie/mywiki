---
type: concept
title: Pluggable Contextual Adapter
slug: pluggable-contextual-adapter
date: 2026-04-20
updated: 2026-04-20
aliases: [PRCA, reward-driven contextual adapter]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pluggable Contextual Adapter** (可插拔上下文适配器) — an intermediate trainable module that rewrites or distills retrieved evidence before it is consumed by a frozen downstream generator.

## Key Points

- [[yang-2023-prca]] introduces PRCA as a plug-in module placed between the retriever and the generator rather than modifying either endpoint directly.
- The adapter is trained in two stages: supervised contextual extraction followed by reward-driven alignment with the downstream generator.
- Its token-level policy is optimized so that the final extracted context preserves useful evidence while shortening the generator input.
- The paper attributes gains in robustness and efficiency to this adapter design, especially on multi-hop and conversational retrieval QA tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-prca]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-prca]].
