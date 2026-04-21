---
type: concept
title: Cold Start
slug: cold-start
date: 2026-04-20
updated: 2026-04-20
aliases: [cold start, warm start initialization]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cold Start** — an initial supervised or otherwise guided phase that gives a policy enough structure to make later optimization stable and productive.

## Key Points

- WebSailor uses a small rejection-sampling fine-tuning stage as a cold start before RL.
- The paper argues that direct RL from an instruction model fails to discover the required long-horizon tool-use patterns reliably.
- The retained cold-start traces are correct, shorter than `32k` tokens, and contain more than five tool calls.
- Ablation results show that the cold-started model converges to stronger final performance than direct RL, especially on BrowseComp-en.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-websailor-2507-02592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-websailor-2507-02592]].
