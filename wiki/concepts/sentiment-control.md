---
type: concept
title: Sentiment Control
slug: sentiment-control
date: 2026-04-20
updated: 2026-04-20
aliases: [sentiment control, 情感控制]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sentiment Control** (情感控制) — conditioning text generation so outputs shift toward desired sentiment polarity or intensity, such as more positive or more negative wording.

## Key Points

- The paper trains sentiment switches on SST-5, using labels `1-2` as negative and `4-5` as positive.
- Positive sentiment generation uses `+5ε_0`, while negative sentiment generation uses `-5ε_0`.
- LM-Switch is not the single best system on every sentiment metric, but the paper reports it remains competitive while using a much simpler mechanism than larger baselines.
- The sentiment experiments also serve as the main empirical demonstration of continuous and compositional control.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
