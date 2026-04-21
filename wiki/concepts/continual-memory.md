---
type: concept
title: Continual Memory
slug: continual-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [continual memory control, memory discipline]
tags: [agents, memory]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Continual Memory** (持续记忆) — an agent memory regime in which memory writes, retrievals, and retention are managed over long task streams while keeping context growth bounded and evaluable.

## Key Points

- ASG-SI treats memory operations as auditable actions inside the policy loop rather than as an external convenience layer.
- The proposed reward decomposition includes a memory-discipline term that penalizes unbounded context growth.
- Periodic replay testing is used to check whether promoted skills and their long-horizon dependencies remain recoverable over time.
- The reference prototype implements a bounded note store to make the control problem concrete.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2025-audited-2512-23760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2025-audited-2512-23760]].
