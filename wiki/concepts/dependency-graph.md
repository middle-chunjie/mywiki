---
type: concept
title: Dependency Graph
slug: dependency-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [Turn Dependency DAG, 依赖图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dependency Graph** (依赖图) — a directed graph that encodes prerequisite relations among elements so valid transformations preserve the dependencies required for correct interpretation.

## Key Points

- ConvAug builds a turn dependency DAG over historical conversational turns by asking an LLM to identify which earlier turns are necessary for understanding each later turn.
- This graph constrains turn masking so ancestor turns of the current query are not removed.
- It also constrains turn reordering by requiring the altered turn order to remain a valid topological ordering of the DAG.
- Dependency-aware operations outperform unconstrained alternatives, showing that structure-preserving augmentation matters for conversational retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-generalizing-2402-07092]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-generalizing-2402-07092]].
