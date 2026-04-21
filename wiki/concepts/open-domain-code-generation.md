---
type: concept
title: Open-Domain Code Generation
slug: open-domain-code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [open domain code generation, 开放域代码生成]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Open-Domain Code Generation** (开放域代码生成) — generating code for natural-language queries that may require arbitrary built-in or third-party libraries rather than only a fixed closed set of operations.

## Key Points

- [[wang-2023-executionbased-2212-10481]] defines open-domain examples as code that uses at least one library and contrasts them with closed-domain examples that use none.
- ODEX contains `505` open-domain samples out of `945` total, spanning `79` libraries and preserving a long-tailed library distribution.
- Open-domain evaluation requires extra execution context such as import prerequisites, function wrapping, and library-specific equality checks.
- The paper shows large accuracy gaps between open-domain and closed-domain subsets for both Codex and CodeGen.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-executionbased-2212-10481]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-executionbased-2212-10481]].
