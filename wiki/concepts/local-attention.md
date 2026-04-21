---
type: concept
title: Local Attention
slug: local-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [localized attention, 局部注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Local Attention** (局部注意力) — an attention pattern that restricts each token to a bounded neighborhood or local segment instead of the full preceding sequence.

## Key Points

- Block Transformer uses local attention in the token decoder, where self-attention is limited to the current block and prefix tokens derived from the block-level context embedding.
- Because previous prompt blocks are never revisited by the token decoder, the model can skip token-decoder prefill for all but the most recent block.
- With main setting `L_B = 4` and prefix length `2`, the token decoder operates on a very small local context while still receiving compressed global information.
- The paper positions this stronger locality against sliding-window approaches, arguing that block-bounded locality across upper layers yields larger KV-cache savings and better prefill behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ho-2024-block-2406-02657]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ho-2024-block-2406-02657]].
