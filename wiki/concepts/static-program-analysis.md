---
type: concept
title: Static Program Analysis
slug: static-program-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [static analysis, 静态程序分析]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Static Program Analysis** (静态程序分析) — reasoning about code properties from source or derived structures without executing the program.

## Key Points

- The paper positions prior code pre-training methods as predominantly static because they learn from source code, ASTs, or related structural views.
- AST-based and code-token backbones remain central in FuzzPretrain; the new contribution is to complement them rather than replace them.
- Static information modeling is instantiated as MLM over code tokens, preserving the standard structure-learning objective used by existing encoders.
- The authors argue that static structure alone struggles to capture functional equivalence across very different implementations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code-2309-09980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code-2309-09980]].
