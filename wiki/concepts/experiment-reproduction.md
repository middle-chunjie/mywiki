---
type: concept
title: Experiment Reproduction
slug: experiment-reproduction
date: 2026-04-20
updated: 2026-04-20
aliases: [实验复现, experimental reproduction]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Experiment Reproduction** (实验复现) — the process of re-running a published computational study to regenerate its analyses, metrics, or artifacts from released code and data.

## Key Points

- AstaBench evaluates reproduction through SUPER-Expert and CORE-Bench-Hard.
- SUPER-Expert focuses on low-resource research repositories that require setup, dependency repair, configuration, execution, and result reporting.
- CORE-Bench-Hard uses CodeOcean capsules and asks agents to perform specific analyses from published papers.
- The benchmark removes GPU-required CORE-Bench-Hard instances to keep resource demands aligned with the rest of the suite.
- Results in the paper show experiment reproduction remains a major bottleneck for current agents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bragg-2026-astabench-2510-21652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bragg-2026-astabench-2510-21652]].
