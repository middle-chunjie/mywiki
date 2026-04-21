---
type: concept
title: Hot Type Slot
slug: hot-type-slot
date: 2026-04-20
updated: 2026-04-20
aliases: [hot slot, 热点类型槽位]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hot Type Slot** (热点类型槽位) — an unresolved variable slot whose type dominates many dependent slots in a type dependency graph and is therefore prioritized for external recommendation.

## Key Points

- HiTYPER computes hot slots after removing already solved nodes from the TDG.
- It applies a dominator algorithm to find unresolved slots that control inference for many other blank slots.
- The paper's rationale is that correctly filling a small number of hot slots lets static inference recover many remaining types.
- Restricting neural queries to hot slots is intended to improve correctness compared with asking a model to predict every blank variable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
