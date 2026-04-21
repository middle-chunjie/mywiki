---
type: concept
title: Collaborative Semantics
slug: collaborative-semantics
date: 2026-04-20
updated: 2026-04-20
aliases: [co-usage semantics, 协作语义]
tags: [agents, representation-learning, tool-use]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Collaborative Semantics** (协作语义) — semantic structure derived from how tools are co-used together in trajectories, rather than from each tool's standalone textual description alone.

## Key Points

- ToolWeaver treats collaborative semantics as the missing signal in flat tool IDs, because rare pairwise co-occurrences make multi-tool reasoning hard to learn.
- The paper constructs a tool-tool similarity matrix from co-usage counts and injects it through a graph-Laplacian regularizer during codebook learning.
- Shared parent codes allow related tools to co-occur densely at higher code levels, which is the main mechanism for improving multi-tool orchestration.
- The strongest empirical gains from collaborative semantics appear on the complex I3 benchmark and on unseen-tool generalization splits.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2026-toolweaver-2601-21947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2026-toolweaver-2601-21947]].
