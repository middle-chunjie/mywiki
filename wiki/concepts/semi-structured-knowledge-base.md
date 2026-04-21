---
type: concept
title: Semi-Structured Knowledge Base
slug: semi-structured-knowledge-base
date: 2026-04-20
updated: 2026-04-20
aliases: [semi-structured kb, semi-structured knowledge base, 半结构化知识库]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semi-Structured Knowledge Base** (半结构化知识库) — a knowledge base that couples explicit entity relations with free-form textual attributes or documents attached to those entities.

## Key Points

- STaRK formalizes an SKB as `G = (V, E)` together with associated text `D = ⋃_{i ∈ V} D_i`.
- The benchmark's answer entities must satisfy both relational constraints from the graph and textual constraints from the linked documents.
- STARK-AMAZON, STARK-MAG, and STARK-PRIME instantiate the concept in e-commerce, academic search, and precision medicine respectively.
- The paper argues that SKBs are a more realistic retrieval setting than text-only or graph-only benchmarks for many production systems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-stark-2404-13207]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-stark-2404-13207]].
