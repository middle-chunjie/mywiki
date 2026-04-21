---
type: concept
title: World Knowledge
slug: world-knowledge
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge recall, 世界知识]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**World Knowledge** (世界知识) — the portion of model capability that depends on memorizing or retrieving factual information about entities, events, and general knowledge from pretraining data.

## Key Points

- [[unknown-nd-code]] measures world knowledge with TriviaQA and Natural Questions Open and reports average exact-match accuracy.
- Among the initialization variants, `balanced -> text` gives the best world-knowledge result, beating text-only by `4.1%` and `code -> text` by `21%`.
- Increasing the code share in from-scratch pretraining consistently harms world-knowledge performance.
- A `100%`-code model suffers an `86%` relative drop in world-knowledge score versus the text-only model because the training mix lacks the necessary factual sources.
- Cooldown with code still helps world knowledge, adding `10.1%` over the no-cooldown model and outperforming cooldown without code.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code]].
