---
type: concept
title: Markov property
slug: markov-property
date: 2026-04-20
updated: 2026-04-20
aliases: [Markov assumption]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Markov property** (马尔可夫性质) — the assumption that the next state of a process depends only on the current state or immediate previous state, not on the full past history.

## Key Points

- The paper's STNM variant models spotlight movement with a Markov update rather than a recurrent history tracker.
- In STNM, the next spotlight handle depends on `s_(t-1)`, the previous spotlight context `sc_(t-1)`, and current decoder state `h_t`.
- This makes spotlight control simpler and easier to train than a full recurrent controller.
- Experimental results show the Markov version is competitive, but the recurrent STNR variant consistently performs better.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yin-2018-transcribing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yin-2018-transcribing]].
