---
type: entity
title: MBCPP
slug: mbcpp
date: 2026-04-20
entity_type: benchmark
aliases: [MBCPP, Multi-lingual Benchmark of C++, C++ code generation benchmark]
tags: []
---

## Description

MBCPP is the C++ function-level code generation benchmark evaluated in [[li-2023-structured-2305-06599]]. The paper describes it as a crowd-sourced benchmark with `848` total problems and `435` test instances used to measure prompting quality in C++.

## Key Contributions

- Extends the paper's evaluation beyond Python benchmarks, showing whether SCoT prompting transfers to another programming language.
- Serves as the benchmark where ChatGPT SCoT reaches `57.06` `Pass@1` and Codex SCoT reaches `48.34` `Pass@1`.

## Related Concepts

- [[code-generation]]
- [[pass-at-k]]
- [[structured-chain-of-thought-prompting]]

## Sources

- [[li-2023-structured-2305-06599]]
