---
type: concept
title: Domain Taxonomy
slug: domain-taxonomy
date: 2026-04-20
updated: 2026-04-20
aliases: [programming domain taxonomy, 领域分类体系]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Domain Taxonomy** (领域分类体系) — a structured inventory of domain categories used to classify tasks or artifacts for analysis and evaluation.

## Key Points

- EvoCodeBench constructs a programming-domain taxonomy from PyPI repository statistics instead of relying only on ad hoc human intuition.
- The taxonomy contains `10` popular domains and is used to label candidate functions during benchmark construction.
- Functions that do not fit any taxonomy category are excluded, which makes domain analysis cleaner but also narrows coverage.
- The first release is imbalanced, with counts such as `120` Scientific Engineering tasks versus only `1` Security task.
- The authors explicitly plan to refine the taxonomy over time as new domains emerge and later benchmark versions grow.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-evocodebench-2410-22821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-evocodebench-2410-22821]].
