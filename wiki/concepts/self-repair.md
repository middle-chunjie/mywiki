---
type: concept
title: Self-Repair
slug: self-repair
date: 2026-04-20
updated: 2026-04-20
aliases: [self repair, 自修复]
tags: [llm, code, debugging]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Repair** (自修复) — a generation strategy in which a model critiques a failing output using execution evidence and then samples revised candidates conditioned on that feedback.

## Key Points

- [[unknown-nd-selfrepair-2306-09896]] models self-repair as a multi-stage process over code generation, execution, feedback generation, and repair.
- The paper argues self-repair should be judged against equal-budget i.i.d. resampling rather than against a zero-cost baseline.
- Performance gains are inconsistent at small budgets and depend strongly on how much budget is spent on diverse initial programs versus repeated repair.
- The study identifies feedback quality, not just code-generation quality, as the main bottleneck limiting self-repair effectiveness.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-selfrepair-2306-09896]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-selfrepair-2306-09896]].
