---
type: concept
title: Atomic Pattern
slug: atomic-pattern
date: 2026-04-20
updated: 2026-04-20
aliases: [AP, 原子模式]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Atomic Pattern** (原子模式) — the smallest indexed adjacency structure in EPR, consisting of either an entity-relation pair or a relation-relation pair used to compose larger evidence patterns.

## Key Points

- The paper decomposes evidence pattern retrieval into ER-APs and RR-APs so that the search space can be indexed and retrieved efficiently.
- RR-APs are serialized with relation labels and link tags, then retrieved from a dense vector index built over millions of candidates.
- Candidate evidence patterns are enumerated by iteratively expanding a partial pattern with atomic patterns that are structurally compatible.
- Retrieval quality of atomic patterns is a major driver of system success; insufficient APs account for `20%` of sampled CWQ errors and `28%` on WebQSP.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-enhancing-2402-02175]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-enhancing-2402-02175]].
