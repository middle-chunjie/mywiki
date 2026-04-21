---
type: concept
title: Compute-Optimal Training
slug: compute-optimal-training
date: 2026-04-20
updated: 2026-04-20
aliases: [Chinchilla-optimal training, 计算最优训练]
tags: [llm, scaling-laws, training]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Compute-Optimal Training** (计算最优训练) — choosing model size and training-token allocation to minimize loss under a fixed compute budget.

## Key Points

- The paper uses compute-optimal supervised scaling laws as the baseline against which all distillation scenarios are compared.
- Teacher and student configurations are analyzed under explicit FLOP constraints, not only under fixed token budgets.
- The work distinguishes four compute-accounting scenarios: amortized teacher, teacher inference, teacher pretraining, and teacher pretraining plus inference.
- When teacher-training cost must be paid for a single student, supervised compute-optimal training remains preferable asymptotically.
- The paper extends the compute-optimal viewpoint from standard pretraining to distillation planning for fixed target student sizes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[busbridge-2025-distillation-2502-08606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[busbridge-2025-distillation-2502-08606]].
