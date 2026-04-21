---
type: concept
title: Difficulty
slug: difficulty
date: 2026-04-20
updated: 2026-04-20
aliases: [难度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Difficulty** (难度) — the latent item parameter that measures how much ability a system needs before it is likely to answer an example correctly.

## Key Points

- In DAD, each item has difficulty `\beta_i`, and larger values mean stronger systems are required to answer the item reliably.
- Difficulty is inferred jointly with subject skill, rather than estimated from raw accuracy alone.
- The paper uses difficulty bins to reveal where top SQuAD systems gain performance: much of the separation comes from harder questions.
- Difficulty alone is not enough for cold-start annotation; selecting only the hardest items performs worse than other IRT-based strategies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[rodriguez-2021-evaluation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[rodriguez-2021-evaluation]].
