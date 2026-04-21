---
type: concept
title: Self-Revision
slug: self-revision
date: 2026-04-20
updated: 2026-04-20
aliases: [iterative self-correction, 自我修订]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Revision** (自我修订) — an inference procedure in which a model regenerates or refines its own outputs over multiple rounds using feedback derived from prior attempts.

## Key Points

- CodeChain performs up to `5` revision rounds after an initial generation round.
- Each revision round conditions on representative sub-modules extracted from previous samples instead of only local feedback from a single candidate.
- The paper reports that APPS performance usually peaks around revision round `4`, with slight degradation by round `5`.
- Harder interview and competition problems benefit more from self-revision than simpler introductory problems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-codechaintowards-2310-08992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-codechaintowards-2310-08992]].
