---
type: concept
title: Context-aware Code Translation
slug: context-aware-code-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [context-aware instruction translation, 上下文感知代码翻译]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Context-aware Code Translation** (上下文感知代码翻译) — translating code into natural-language-like descriptions while explicitly incorporating execution context such as variables, constants, data dependencies, and control dependencies.

## Key Points

- TranCS first disassembles Java code into instructions and then translates them into short natural-language sentences.
- The translation fills placeholders with stack values, local variables, constants, and branch targets recovered during static simulation.
- This design is intended to preserve semantics better than structural representations like ASTs or CFGs alone.
- The produced translations are closer to developer queries than raw code tokens, reducing modality mismatch in code search.
- Ablation results show context-aware translation improves retrieval over token-only baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-code-2202-08029]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-code-2202-08029]].
