---
type: concept
title: No position encoding
slug: no-position-encoding
date: 2026-04-20
updated: 2026-04-20
aliases: [NoPE]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**No position encoding** — a design choice that omits explicit positional encodings in an attention module and relies on other architectural components to represent order and recency.

## Key Points

- Kimi Linear applies NoPE to all full MLA layers and pushes positional modeling into KDA.
- The paper argues this yields a more balanced positional bias across depth than combining strong RoPE with weaker linear-attention inductive bias.
- NoPE also lets MLA convert to a more efficient MQA-style inference form and avoids long-context retuning of RoPE frequencies.
- In long-context evaluation, Kimi Linear with NoPE outperforms the RoPE variant on average (`54.5` vs. `51.8`).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[team-2025-kimi-2510-26692]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[team-2025-kimi-2510-26692]].
