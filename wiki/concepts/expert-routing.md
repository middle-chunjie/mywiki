---
type: concept
title: Expert Routing
slug: expert-routing
date: 2026-04-20
updated: 2026-04-20
aliases: [routing, token routing, 专家路由]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Expert Routing** (专家路由) — the gating process in a Mixture-of-Experts model that selects and weights specialized experts for a given token or input.

## Key Points

- The paper models routing at layer `l` as a softmax gate over expert logits, producing a probability vector over `N^(l)` experts.
- These routing weights encode how the model internally decomposes an input across specialized experts, rather than only what token it predicts next.
- Concatenating routing decisions across layers yields a representation that captures semantic structure at multiple depths of the network.
- The authors show that routing-derived signals are more robust to prompt variation than hidden-state embeddings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-your-2410-10814]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-your-2410-10814]].
