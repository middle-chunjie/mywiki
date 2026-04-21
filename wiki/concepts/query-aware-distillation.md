---
type: concept
title: Query-Aware Distillation
slug: query-aware-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [query-aware passage token distillation, 查询感知蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Query-Aware Distillation** (查询感知蒸馏) — a distillation objective that supervises only the passage tokens most relevant to the query instead of all cached passage tokens equally.

## Key Points

- [[unknown-nd-btrbinary-2310-01329]] introduces this loss for the decomposed reader before binarization to reduce the accuracy loss from separating query and passage encoding.
- The teacher is the original non-decomposed reader, while the student is the decomposed reader that will later initialize BTR.
- Salient passage tokens are selected by query-to-passage attention, and the paper sets `r = 50%` of total passage tokens by default.
- The loss is written as `` `L_distill = (1/r) Σ_i (h_i - h_i^decomp)^2` `` over only the selected tokens.
- Removing this objective lowers NaturalQuestions accuracy from `49.5` to `48.2` in the paper's ablation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-btrbinary-2310-01329]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-btrbinary-2310-01329]].
