---
type: concept
title: Multilingual Code Generation
slug: multilingual-code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [multilingual NL-to-code generation, 多语言代码生成]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Code Generation** (多语言代码生成) — generating code from natural-language instructions written in multiple human languages while preserving task semantics across languages.

## Key Points

- [[wang-2023-executionbased-2212-10481]] includes intents in English, Spanish, Japanese, and Russian, with `439`, `90`, `164`, and `252` samples respectively.
- The benchmark reuses Stack Overflow-derived supervision and adds execution-based tests for all four languages.
- Codex and CodeGen both show non-trivial multilingual capability even though neither family was explicitly designed for multilingual usage in this study.
- The paper reports that Spanish examples have longer code on average, while English examples have fewer input and output variables, indicating language-conditioned difficulty differences.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-executionbased-2212-10481]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-executionbased-2212-10481]].
