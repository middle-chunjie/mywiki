---
type: concept
title: End-to-End Discovery
slug: end-to-end-discovery
date: 2026-04-20
updated: 2026-04-20
aliases: [端到端发现, automated scientific discovery]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**End-to-End Discovery** (端到端发现) — the execution of a full research workflow by an agent, from planning and experiment design to implementation, analysis, and final reporting.

## Key Points

- AstaBench evaluates this capability with E2E-Bench and E2E-Bench-Hard.
- Task inputs specify an AI/NLP research question plus a detailed sequence of steps to investigate it.
- Outputs are judged across report quality, generated code, and produced artifacts against example-specific rubrics.
- Because these tasks are expensive, the framework supports cache-based agents whose outputs are precomputed offline.
- The paper shows that partial step completion can look decent while full experiment completion remains extremely rare.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bragg-2026-astabench-2510-21652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bragg-2026-astabench-2510-21652]].
