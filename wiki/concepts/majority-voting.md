---
type: concept
title: Majority Voting
slug: majority-voting
date: 2026-04-20
updated: 2026-04-20
aliases: [majority vote, 多数投票]
tags: [aggregation, reasoning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Majority Voting** (多数投票) — an aggregation rule that selects the answer receiving the highest count among multiple candidate outputs.

## Key Points

- Self-consistency operationalizes majority voting as `arg max_a Σ_i 1(a_i = a)` over final answers parsed from sampled reasoning traces.
- On PaLM-540B, unweighted majority vote performs nearly identically to normalized weighted sum and clearly better than weighted averages.
- Table 1 reports `74.4` on GSM8K with majority voting versus `56.5` for greedy decoding and only `22.1` for normalized weighted average.
- The paper treats agreement among independent sampled paths as a practical signal of answer reliability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-selfconsistency-2203-11171]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-selfconsistency-2203-11171]].
