---
type: concept
title: Organizer-Worker Architecture
slug: organizer-worker-architecture
date: 2026-04-20
updated: 2026-04-20
aliases: [organizer-workers architecture, planner-executor architecture, 组织者-工作者架构]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Organizer-Worker Architecture** (组织者-工作者架构) — a multi-agent design that separates high-level planning and constraint reasoning from low-level tool execution and evidence collection.

## Key Points

- InfoMosaic-Flow uses an organizer to plan and a worker to execute domain tool calls and return consolidated evidence.
- The separation is intended to preserve reasoning depth while reducing execution noise during multi-step synthesis.
- The architecture also enlarges the search space because the organizer remains tool-agnostic while the worker explores the full domain toolset.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2026-infomosaicbench-2510-02271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2026-infomosaicbench-2510-02271]].
