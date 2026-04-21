---
type: entity
title: AdvTest
slug: advtest
date: 2026-04-20
entity_type: dataset
aliases: [AdvTest, AdvTest benchmark]
tags: [benchmark, code-search, robustness]
---

## Description

AdvTest is a robustness-focused NL2Code search benchmark derived from CodeSearchNet Python functions with identifiers (function and variable names) replaced by dummy variables. It tests whether a model understands code semantics rather than relying on surface-level lexical cues.

## Key Contributions

- Stresses code retrieval models beyond lexical matching by removing informative identifier names.
- Introduced in CodeXGLUE as part of the code search evaluation suite.
- Used in [[zhang-2024-code-2402-01935]] to demonstrate CodeSage's generalization; CodeSage-large achieves MRR 52.67%, the best reported result.

## Related Concepts

- [[code-search]]
- [[code-representation-learning]]

## Sources

- [[zhang-2024-code-2402-01935]]
