---
type: concept
title: Budget Forcing
slug: budget-forcing
date: 2026-04-20
updated: 2026-04-20
aliases: [S1 budget forcing]
tags: [llm, inference]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Budget Forcing** — a test-time control strategy that tries to enforce a token budget by truncating generation or injecting special continuation tokens when the model is too short or too long.

## Key Points

- The paper uses S1 as the representative budget-forcing baseline for reasoning models.
- S1 inserts control tokens such as "Wait" or "Final Answer" after hitting a budget threshold.
- The authors argue that budget forcing interrupts reasoning mid-step and leads to brittle, hand-engineered behavior.
- On the paper's math benchmarks, LCPO clearly outperforms this strategy at the same token budgets.
- Budget forcing is presented as a decoding-time heuristic, not a training-time adaptation of reasoning behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
