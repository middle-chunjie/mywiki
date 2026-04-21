---
type: concept
title: Hybrid Type Inference
slug: hybrid-type-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [hybrid typing, 混合类型推断]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hybrid Type Inference** (混合类型推断) — a type inference strategy that combines rule-based static reasoning with learned type recommendations under an explicit validation loop.

## Key Points

- HiTYPER uses static inference as the authoritative mechanism and only asks neural models for unresolved type slots.
- Neural suggestions are not trusted directly; they are filtered by rejection rules and then re-used by another static inference pass.
- The hybrid design targets the complementary strengths of static precision and deep-learning flexibility on underconstrained variables.
- The paper reports that this combination improves both overall accuracy and rare-type prediction compared with purely neural baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
