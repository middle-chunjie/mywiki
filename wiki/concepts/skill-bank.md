---
type: concept
title: Skill Bank
slug: skill-bank
date: 2026-04-20
updated: 2026-04-20
aliases: [技能库]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Skill Bank** (技能库) — an external repository of reusable natural-language skills that can be retrieved and injected into an agent's decision context.

## Key Points

- D2Skill divides the bank into task skills for high-level guidance and step skills for local correction.
- Skills are created from training-time reflection rather than harvested from validation trajectories.
- Each skill carries a retrieval key and a utility estimate that is updated online from rollout outcomes.
- The bank is dynamically expanded, queried, and pruned so that memory quality does not degrade as training proceeds.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
