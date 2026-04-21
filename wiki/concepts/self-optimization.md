---
type: concept
title: Self-Optimization
slug: self-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [self optimization, 自优化]
tags: [llm, code-generation, optimization]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Self-Optimization** (自优化) — a procedure where a model revises its own generated artifact using feedback from execution or evaluation signals to improve a target property such as efficiency.

## Key Points

- [[huang-2024-effilearner-2405-15189]] applies self-optimization after the model has already produced code that passes open tests.
- The refinement signal is not a scalar reward alone, but line-level runtime and memory profiles tied to the generated program.
- The process is iterative, with ablations over `0` to `5` steps showing large first-step gains and diminishing returns afterward.
- The target is joint runtime and memory reduction while retaining [[functional-correctness]].

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2024-effilearner-2405-15189]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2024-effilearner-2405-15189]].
