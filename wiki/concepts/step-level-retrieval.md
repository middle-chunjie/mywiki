---
type: concept
title: Step-Level Retrieval
slug: step-level-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [step-level experience retrieval, per-step retrieval, 步级检索]
tags: [agents, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Step-Level Retrieval** (步级检索) — a retrieval scheme that fetches supporting experiences or context at each decision step using the agent's current observation rather than a single static task description.

## Key Points

- SLEA-RL retrieves experiences `\varepsilon_t` at every step `t`, conditioning retrieval on the current observation instead of only on the initial task prompt.
- Retrieval first uses the current observation cluster and falls back to library-wide matching only when the cluster has no associated experiences.
- The implementation retrieves top `2` strategy entries and top `1` warning entry per step after a `5`-epoch warmup.
- The paper attributes gains over task-level retrieval on ALFWorld and WebShop to this dynamic conditioning as the environment state evolves through long-horizon interaction.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2026-slearl-2603-18079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2026-slearl-2603-18079]].
