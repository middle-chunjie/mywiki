---
type: concept
title: Multi-Hop Reasoning
slug: multi-hop-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [multi hop reasoning, 多跳推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-Hop Reasoning** (多跳推理) — reasoning that requires combining evidence from two or more distinct context segments rather than extracting a single local fact.

## Key Points

- One branch of In2 training explicitly generates QA pairs whose answers depend on at least two `~128`-token segments sampled from the same source text.
- Required segments are jointly shuffled with distractors in the long context, so the model must recover and integrate distant evidence rather than rely on local cues.
- The resulting model shows large gains on multi-document and multi-hop tasks such as HotpotQA (`42.4 -> 62.1`), 2WikiMQA (`24.3 -> 47.0`), and MuSiQue (`20.8 -> 39.0`).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[an-2024-make-2404-16811]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[an-2024-make-2404-16811]].
