---
type: concept
title: Lost-in-the-Middle
slug: lost-in-the-middle
date: 2026-04-20
updated: 2026-04-20
aliases: [lost in the middle, 长上下文中部信息丢失]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Lost-in-the-Middle** (长上下文中部信息丢失) — the tendency of a long-context model to underuse relevant evidence located in the middle of the prompt compared with evidence near the beginning or end.

## Key Points

- The paper treats lost-in-the-middle as the main failure mode preventing nominal `32K` models from fully exploiting long contexts.
- On the paper's VaL probing suite, Mistral-7B-Instruct-v0.2 shows large position sensitivity, while FilM-7B substantially flattens the performance curve across context positions.
- The authors argue that near-perfect Needle-in-the-Haystack scores can mask this problem because that task uses familiar document context and an easier forward-retrieval pattern.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[an-2024-make-2404-16811]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[an-2024-make-2404-16811]].
