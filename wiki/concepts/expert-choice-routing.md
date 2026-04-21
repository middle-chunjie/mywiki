---
type: concept
title: Expert-Choice Routing
slug: expert-choice-routing
date: 2026-04-20
updated: 2026-04-20
aliases: [expert choice, 专家选择路由]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Expert-Choice Routing** (专家选择路由) — a sparse MoE routing scheme in which experts select tokens, aiming to balance expert load without auxiliary load-balancing losses.

## Key Points

- The paper uses expert-choice routing in all MoE experiments rather than standard token-choice top-k routing.
- Tokens are grouped by position across sequences, with group size `256`, to make expert choice compatible with autoregressive language modeling.
- The authors apply softmax over experts and select tokens over the token dimension because that performed best in their setup.
- Expert-choice routing is important to the paper's scaling claims because it keeps expert load balanced while granularity increases.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[krajewski-2024-scaling-2402-07871]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[krajewski-2024-scaling-2402-07871]].
