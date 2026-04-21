---
type: concept
title: Budget Control
slug: budget-control
date: 2026-04-20
updated: 2026-04-20
aliases: [budget controller, adaptive budget allocation, 预算控制]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Budget Control** (预算控制) — a mechanism that allocates different compression budgets across prompt components so a global token target is met without compressing all parts equally.

## Key Points

- [[jiang-2023-llmlingua-2310-05736]] assigns more budget to instructions and the question, and less to demonstrations, because prompt components have different sensitivity.
- The demonstration compression rate is computed analytically as `τ_dems = (τL - (τ_ins L_ins + τ_que L_que)) / L_dems`.
- The controller ranks demonstrations by perplexity from the small LM and keeps them until the selected set would exceed `k · τ_dems · L_dems`.
- Remaining unused budget is redistributed to instruction and question through `Δτ`, so compression decisions are coupled across prompt sections.
- Ablation in the paper shows removing the budget controller lowers GSM8K `1-shot` EM from `79.08` to `73.62`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2023-llmlingua-2310-05736]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2023-llmlingua-2310-05736]].
