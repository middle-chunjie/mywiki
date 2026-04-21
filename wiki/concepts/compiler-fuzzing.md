---
type: concept
title: Compiler Fuzzing
slug: compiler-fuzzing
date: 2026-04-20
updated: 2026-04-20
aliases: [编译器模糊测试]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Compiler fuzzing** (编译器模糊测试) — the practice of generating and executing programs to expose compiler crashes, hangs, miscompilations, or robustness failures.

## Key Points

- The paper treats compiler fuzzing as a structured-input generation problem because invalid programs are rejected early and do not exercise deeper compiler logic.
- ALPHAPROG avoids handwritten grammars and instead learns to synthesize Brainfuck programs directly from compiler feedback.
- Validity and diversity are both core objectives: valid programs must compile, while diverse programs should cover new compiler basic blocks.
- The environment signal includes compilation messages, execution traces, and, in the strongest reward, program cyclomatic complexity.
- On BFC, the learned generator outperforms AFL in both valid-input production and accumulated coverage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2022-alphaprog]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2022-alphaprog]].
