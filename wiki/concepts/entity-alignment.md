---
type: concept
title: Entity Alignment
slug: entity-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [实体对齐, knowledge graph entity alignment]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Entity Alignment** (实体对齐) — the task of identifying equivalent entities across different knowledge graphs, typically under partial supervision from a seed alignment set.

## Key Points

- RHGN formulates entity alignment between two KGs `G1` and `G2` as recovering a 1-to-1 alignment relation over entities.
- The paper emphasizes that neighbor heterogeneity and relation heterogeneity are two distinct sources of alignment difficulty.
- RHGN uses relation-gated message passing so relation signals control information flow instead of being directly merged with entity embeddings.
- The model also exchanges aligned entities across graphs to shorten the path by which evidence propagates between KGs.
- Training uses a contrastive alignment loss with hard negatives and a separate relation alignment objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-rhgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-rhgn]].
