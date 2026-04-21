---
type: entity
title: HumanEval
slug: humaneval
date: 2026-04-20
entity_type: tool
aliases: [HumanEval]
tags: [benchmark, code, evaluation]
---

## Description

HumanEval is a manually curated Python code-generation benchmark used in [[allamanis-2024-unsupervised-2402-08699]] as a narrow-domain reference for validating [[round-trip-correctness]].

## Key Contributions

- Provides the standard `pass@1` baseline against which RTC is compared in the paper.
- Supplies unit-test oracles that make execution-based synthesis evaluation possible.

## Related Concepts

- [[code-synthesis]]
- [[execution-based-evaluation]]
- [[pass-at-k]]

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]
