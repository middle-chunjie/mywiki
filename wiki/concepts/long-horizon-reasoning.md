---
type: concept
title: Long-Horizon Reasoning
slug: long-horizon-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [extended-horizon reasoning, long-horizon tasks]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Horizon Reasoning** (长时程推理) — reasoning over many sequential interactions where performance depends on sustained planning, evidence integration, and error control across extended trajectories.

## Key Points

- The paper treats deep research as a long-horizon setting because difficult questions require many cycles of search, reading, synthesis, and revision.
- Mono-contextual agents are argued to fail on long horizons due to shrinking reasoning space and irrecoverable accumulation of noisy evidence.
- IterResearch addresses the problem by reconstructing a bounded workspace at each step instead of preserving full history.
- The empirical study spans benchmarks such as BrowseComp and GAIA, where many interactions may be necessary before answering.
- The scaling experiment shows that performance can continue improving as the allowed interaction horizon increases to `2048` rounds.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
