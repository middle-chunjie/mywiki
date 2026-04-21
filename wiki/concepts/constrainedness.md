---
type: concept
title: Constrainedness
slug: constrainedness
date: 2026-04-20
updated: 2026-04-20
aliases: [query constrainedness]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Constrainedness** (约束度) — a scalar measure of how restrictive a query is, commonly increasing as the fraction of valid solutions decreases.

## Key Points

- KITAB defines constrainedness as `kappa = 1 - S / N` where `S` is the number of satisfying books and `N` is the total number of books for the author.
- The dataset records constrainedness to support controlled analysis of how query difficulty changes across lexical, temporal, and entity-based constraints.
- For many constraint types, higher constrainedness correlates with worse LLM performance, but the paper reports exceptions such as `ends-with` and `city-name`.
- Aggregated results can look bimodal because different constraint families occupy very different constrainedness ranges and failure modes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-kitabevaluating-2310-15511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-kitabevaluating-2310-15511]].
