---
type: concept
title: Evolving Memory
slug: evolving-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [shared memory, 演化记忆]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Evolving Memory** (演化记忆) — a deterministic structured state that is updated after each turn to record actions, tool outputs, verification signals, and other context needed for future decisions.

## Key Points

- AgentFlow initializes `M^1` from the user query and updates memory after each turn with `f_mem(M^t, a^t, e^t, v^t)`.
- The memory stores concise process information rather than preserving unbounded raw history.
- The paper uses memory to keep planning transparent, controllable, and bounded as the interaction horizon grows.
- Appendix E describes regex-based extraction of planner outputs, tool commands, results, and verifier status into the shared memory record.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-intheflow-2510-05592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-intheflow-2510-05592]].
