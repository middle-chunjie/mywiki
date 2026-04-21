---
type: concept
title: Quality Check
slug: quality-check
date: 2026-04-20
updated: 2026-04-20
aliases: [quality check, 质量检查]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Quality Check** (质量检查) — a filtering procedure that evaluates generated candidates against explicit criteria before they are retained for further use.

## Key Points

- EvoAgent inserts an LLM-based quality-check module into the selection stage to retain agents that are both competent and sufficiently distinct from their parents.
- The paper's ablations show that quality check matters more when population size grows, because otherwise multiple generated agents can collapse to near-duplicates.
- On TravelPlanner, the quality-check module reduces failures caused by unsuitable specialists that over-focus on narrow expertise and produce unusably long plans.
- The selection stage is therefore not a cosmetic add-on; it is part of the paper's argument that automatic agent generation must control diversity and usefulness simultaneously.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yuan-2024-evoagent-2406-14228]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yuan-2024-evoagent-2406-14228]].
