---
type: concept
title: Skill Graph
slug: skill-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [skill graph, audited skill graph]
tags: [agents, modularity]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Skill Graph** (技能图) — a directed graph of reusable skills whose nodes carry explicit interfaces and verification reports, while edges encode composition constraints, ordering, and guarded fallback behavior.

## Key Points

- In ASG-SI, promoted skills are stored as nodes instead of remaining implicit in model weights.
- Edge structure makes composition constraints explicit and supports failure localization at the node or edge level.
- The graph is meant to turn capability growth into an accumulative, inspectable asset rather than a benchmark-only score increase.
- Verified skills remain available as stable fallbacks even if the base policy later drifts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2025-audited-2512-23760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2025-audited-2512-23760]].
