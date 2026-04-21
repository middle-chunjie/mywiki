---
type: concept
title: Cold-Start Recommendation
slug: cold-start-recommendation
date: 2026-04-20
updated: 2026-04-20
aliases: [cold start, 冷启动推荐]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cold-Start Recommendation** (冷启动推荐) — recommendation under sparse or absent historical interactions, where user or item representations must rely on auxiliary signals beyond dense feedback logs.

## Key Points

- [[jin-2023-code]] defines cold-start users as developers with `<= 2` training interactions.
- CODER addresses cold start by combining file semantics, project hierarchy, and macroscopic user-project behaviors.
- On cold-start benchmarks, the model reports especially large relative gains, including `54.0%` improvement on DB `MRR@3`.
- The paper argues that macro-level repository interests are particularly useful when file-level contribution histories are sparse.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2023-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2023-code]].
