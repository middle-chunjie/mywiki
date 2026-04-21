---
type: concept
title: Pivotal Token Search
slug: pivotal-token-search
date: 2026-04-20
updated: 2026-04-20
aliases: [PTS, pivotal token search]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pivotal Token Search** — a token-level preference-data construction method that finds single-token decision points whose inclusion materially changes the probability that a completion succeeds.

## Key Points

- PTS recursively subdivides a completion until a segment is either length `1` or changes estimated success probability by less than a threshold `p_gap`.
- The resulting DPO pair focuses on the exact token where a trajectory becomes more or less successful, rather than comparing only whole divergent completions.
- Phi-4 applies PTS to math, question answering, and coding tasks where correctness can be checked with an oracle or test suite.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
