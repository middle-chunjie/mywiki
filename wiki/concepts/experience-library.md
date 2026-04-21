---
type: concept
title: Experience Library
slug: experience-library
date: 2026-04-20
updated: 2026-04-20
aliases: [experience memory, episodic experience library, 经验库]
tags: [agents, memory]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Experience Library** (经验库) — a maintained repository of reusable strategies and warnings distilled from prior trajectories and retrieved to guide future agent decisions.

## Key Points

- In SLEA-RL the library is split into a strategy zone `E^{+}` and a warning zone `E^{-}`, and each entry carries an abstraction level plus a trajectory-derived quality score.
- Library updates use top/bottom trajectory selection, LLM-based semantic extraction, and score-based admission rather than direct gradient updates on the memory itself.
- The default configuration uses per-level capacities of `100` strategy entries and `50` warning entries, with rate limits on how many new items may be added per update.
- Retrieved experiences are reused during both training and inference, so the policy learns to act under the same memory-augmented prompting regime that it sees at deployment.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-slearl-2603-18079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-slearl-2603-18079]].
