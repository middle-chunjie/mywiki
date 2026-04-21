---
type: concept
title: Agent Data Protocol
slug: agent-data-protocol
date: 2026-04-20
updated: 2026-04-20
aliases: [ADP, 智能体数据协议]
tags: [agents, data]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agent Data Protocol** (智能体数据协议) — a standardized schema for representing agent trajectories as typed actions and observations so heterogeneous datasets can be converted into reusable training pipelines.

## Key Points

- ADP treats each example as a `Trajectory` with an alternating action-observation `content` sequence plus flexible dataset-specific `details`.
- It supports three action types (`APIAction`, `CodeAction`, `MessageAction`) and two observation types (`TextObservation`, `WebObservation`).
- The paper uses ADP to unify `13` public datasets into a single `1.3M`-trajectory corpus for agent supervised fine-tuning.
- ADP enables one `Raw -> ADP` converter per dataset and one `ADP -> SFT` converter per harness, replacing pairwise dataset-harness engineering.
- The protocol is paired with automated validation for tool formatting, thought coverage, and conversation integrity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[song-2026-agent-2510-24702]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[song-2026-agent-2510-24702]].
