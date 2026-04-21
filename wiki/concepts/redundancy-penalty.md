---
type: concept
title: Redundancy Penalty
slug: redundancy-penalty
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval redundancy penalty, 冗余惩罚]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Redundancy Penalty** (冗余惩罚) — a process-level penalty that discourages search actions whose retrieved evidence substantially overlaps with documents already seen in earlier steps.

## Key Points

- StepSearch keeps a cumulative retrieval history `H^t` and penalizes current-round documents that already appeared in `H^(t-1)`.
- The penalty is defined as `P^t = (1 / k) * sum_j 1(d_j^{r(t)} in H^(t-1))`, i.e. the fraction of repeated documents in the current retrieval set.
- Ablation results show that redundancy penalty alone is insufficient, but combined with information gain it raises the search quality ceiling.
- The paper motivates this term by observing that repeated confirmatory queries waste search budget and amplify hallucination risk.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-stepsearch-2505-15107]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-stepsearch-2505-15107]].
