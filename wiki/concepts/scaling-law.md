---
type: concept
title: Scaling Law
slug: scaling-law
date: 2026-04-20
updated: 2026-04-20
aliases: [scaling laws, 扩展定律]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Scaling Law** (扩展定律) — an empirical regularity that relates model performance or loss to resources such as parameters, data, or compute through a predictable functional form, often a power law.

## Key Points

- This paper re-estimates LLM scaling laws under a fixed compute budget instead of assuming training-token count is approximately constant.
- Across three estimation methods, the paper finds that optimal parameter count and optimal token count both scale with compute at exponents near `0.5`.
- The analysis directly challenges the Kaplan et al. prescription that parameters should scale much faster than data.
- A parametric loss model of the form `` `L̂(N,D) = E + A / N^α + B / D^β` `` is used to make the resource trade-off explicit.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hoffmann-2022-training-2203-15556]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hoffmann-2022-training-2203-15556]].
