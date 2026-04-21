---
type: concept
title: Interaction Scaling
slug: interaction-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [interaction budget scaling]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Interaction Scaling** (交互扩展性) — the ability of an agent to preserve or improve performance as the allowed number of tool-use and reasoning rounds increases substantially.

## Key Points

- The paper frames interaction scaling as a core differentiator between iterative and mono-contextual research agents.
- IterResearch reports BrowseComp-200 accuracy improvements as the interaction budget grows from `2` to `2048` turns.
- Despite a very high budget, actual average usage stays far lower (`80.1` turns under the `2048`-turn cap), suggesting adaptive stopping.
- The authors argue that constant-structure workspace reconstruction, rather than merely larger context windows, makes this scaling behavior possible.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
