---
type: concept
title: Data-Driven Discovery
slug: data-driven-discovery
date: 2026-04-20
updated: 2026-04-20
aliases: [数据驱动发现, hypothesis discovery]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Data-Driven Discovery** (数据驱动发现) — the task of deriving and verifying specific hypotheses from data by identifying relevant context, variables, and relationships.

## Key Points

- AstaBench operationalizes this setting through DiscoveryBench.
- Each task provides one or more datasets plus a discovery goal, and the agent must produce a specific supported hypothesis.
- The paper formalizes a hypothesis with three facets: context, variables, and relationship.
- DiscoveryBench spans `6` domains, including sociology, biology, humanities, economics, engineering, and meta-science.
- Evaluation is based on judge-model alignment between predicted and gold hypotheses, not exact lexical matching.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bragg-2026-astabench-2510-21652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bragg-2026-astabench-2510-21652]].
