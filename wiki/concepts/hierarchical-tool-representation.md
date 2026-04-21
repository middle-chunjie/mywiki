---
type: concept
title: Hierarchical Tool Representation
slug: hierarchical-tool-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [hierarchical tool code, 分层工具表示]
tags: [agents, representation-learning, tool-use]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Tool Representation** (分层工具表示) — representing each tool with a multi-level code sequence so high-level codes capture shared structure and lower-level codes disambiguate individual tools.

## Key Points

- ToolWeaver maps each tool to a sequence like `[iota_1, ..., iota_L]` rather than to a single special token.
- The representation capacity scales as `K^L`, while vocabulary growth is only `L x K`.
- Shared parent codes are meant to reflect broad functional or collaborative groupings, which the paper argues are useful for multi-step planning.
- Compared with numerical or static hierarchical baselines, ToolWeaver's learned hierarchy is guided by both semantics and co-usage statistics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
