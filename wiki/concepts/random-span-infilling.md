---
type: concept
title: Random Span Infilling
slug: random-span-infilling
date: 2026-04-20
updated: 2026-04-20
aliases: [random span infill, 随机跨度填充]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Random Span Infilling** (随机跨度填充) — an infilling evaluation setup in which the missing span is sampled uniformly at random from positions inside a reference solution.

## Key Points

- The paper introduces this benchmark to complement single-line and multi-line infilling tasks derived from HumanEval solutions.
- The full benchmark contains `1640` tasks, while a light version with `164` tasks is used to monitor training dynamics.
- Random-span evaluation stresses cases where the removed span starts or ends inside a token, making it more realistic than line-only masking.
- Character-level FIM training yields much stronger results on this benchmark than token-level or line-level span selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[bavarian-2022-efficient-2207-14255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[bavarian-2022-efficient-2207-14255]].
