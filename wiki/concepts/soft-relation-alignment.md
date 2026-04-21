---
type: concept
title: Soft Relation Alignment
slug: soft-relation-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [软关系对齐, SRA]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Soft Relation Alignment** (软关系对齐) — a relation matching strategy that generates multi-label supervision from entity-derived relation prototypes instead of requiring gold relation correspondences across graphs.

## Key Points

- RHGN represents each relation by concatenating the mean embedding of its head entities and tail entities.
- Soft labels are induced by thresholding cosine similarity between relation label embeddings with `γ = 0.5`.
- The paper explicitly models relation alignment as multi-label rather than 1-to-1 because one relation can correspond to multiple relations or none.
- SRA contributes complementary gains to entity alignment by reducing relation heterogeneity between graphs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-rhgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-rhgn]].
