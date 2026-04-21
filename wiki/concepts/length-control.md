---
type: concept
title: Length Control
slug: length-control
date: 2026-04-20
updated: 2026-04-20
aliases: [output-length control, token-budget adherence]
tags: [llm, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Length Control** (长度控制) — the ability of a generative model to produce outputs whose token count closely matches an exact target or stays under a requested maximum budget.

## Key Points

- This paper treats output length as a controllable deployment variable rather than a side effect of decoding.
- L1 is evaluated at target budgets `{512, 1024, 2048, 3600}` and can smoothly trade off accuracy against token usage.
- On math datasets, L1 reports about `3%` mean length error, showing that RL can teach token-budget adherence with high precision.
- OOD adherence is weaker, with roughly `20-40%` error on general reasoning and knowledge tasks.
- For maximum-budget mode, soft violation rates remain low at `0.3-2.3%`, suggesting practical reliability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
