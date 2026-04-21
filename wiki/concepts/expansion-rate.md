---
type: concept
title: Expansion Rate
slug: expansion-rate
date: 2026-04-20
updated: 2026-04-20
aliases: [MoE expansion rate, 扩展率]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Expansion Rate** (扩展率) — the ratio `E = N_MoE / N_ff` measuring how many total MoE parameters are available relative to the dense feed-forward layer they replace.

## Key Points

- The paper uses expansion rate to separate total MoE capacity from the number of active parameters used per token.
- The number of experts is expressed as `N_expert = G * E`, linking expansion rate directly to granularity.
- When `G = 1`, the expansion rate reduces to the usual interpretation of the number of experts.
- The main empirical sweep fixes `E = 64` to isolate the effect of model size, token count, and granularity.
- Appendix experiments with `E = 16` show similar trends but with wider uncertainty intervals.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[krajewski-2024-scaling-2402-07871]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[krajewski-2024-scaling-2402-07871]].
