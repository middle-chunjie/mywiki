---
type: entity
title: ODEX
slug: odex
date: 2026-04-20
entity_type: dataset
aliases: [Open-Domain EXecution-based dataset]
tags: []
---

## Description

ODEX is the benchmark dataset introduced in [[wang-2023-executionbased-2212-10481]] for execution-based evaluation of open-domain natural-language-to-Python code generation. It pairs Stack Overflow-derived NL-code examples with executable human-written tests.

## Key Contributions

- Provides `945` examples, `1,707` test cases, and coverage of `79` libraries across four natural languages.
- Makes open-domain code generation executable by attaching function wrappers, library prerequisites, and task-specific assertions.
- Reveals distinct open-versus-closed domain behaviors for Codex and CodeGen.

## Related Concepts

- [[code-generation-benchmark]]
- [[execution-based-evaluation]]
- [[open-domain-code-generation]]

## Sources

- [[wang-2023-executionbased-2212-10481]]
