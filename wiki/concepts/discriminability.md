---
type: concept
title: Discriminability
slug: discriminability
date: 2026-04-20
updated: 2026-04-20
aliases: [discrimination, 区分度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Discriminability** (区分度) — the item parameter that measures how strongly an example separates higher-skill systems from lower-skill systems.

## Key Points

- DAD models discriminability with `\gamma_i`, which scales the skill-minus-difficulty gap in the response function.
- High-discriminability examples are especially useful for ranking because stronger systems are more likely to answer them correctly while weaker ones fail.
- Negative discriminability is a red flag: it means weaker systems are more likely than stronger ones to get an item right.
- The paper shows that items with negative or near-zero discriminability often correspond to annotation problems or otherwise flawed evaluation examples.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
