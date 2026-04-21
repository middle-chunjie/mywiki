---
type: concept
title: Safety Alignment
slug: safety-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [safety alignment, 安全对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Safety Alignment** (安全对齐) — post-pretraining adaptation that steers a model to follow safety policies and human value constraints when responding to user requests.

## Key Points

- The paper treats refusal behavior as an observable outcome of safety alignment in deployed LLMs.
- It frames current practice as combining pretraining with methods such as instruction tuning and preference optimization.
- SORRY-Bench is designed to measure where aligned models still fulfill unsafe requests despite those safeguards.
- The results show that stronger refusal is not uniform across categories, prompt forms, or model versions, indicating alignment remains brittle.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2024-sorrybench-2406-14598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2024-sorrybench-2406-14598]].
