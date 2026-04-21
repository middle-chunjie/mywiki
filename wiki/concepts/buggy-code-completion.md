---
type: concept
title: Buggy-Code Completion
slug: buggy-code-completion
date: 2026-04-20
updated: 2026-04-20
aliases: [bCC, buggy code completion, 带潜在缺陷代码补全]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Buggy-Code Completion** (带潜在缺陷代码补全) — the task of generating a functionally correct program from a specification and a partial code prefix that may contain potential semantic bugs.

## Key Points

- [[dinh-2023-large-2306-03438]] formalizes bCC as completion from a buggy prefix `s'` without requiring the final output to preserve that prefix exactly.
- The paper argues bCC is not reducible to standard program repair followed by completion because bugginess is defined with respect to possible completions of partial code.
- Two benchmarks are introduced for bCC: synthetic-operator buggy-HumanEval and realistic-submission buggy-FixEval.
- Existing code LLMs fail dramatically on this task, with pass@1 dropping to low single digits under buggy prefixes.
- The study evaluates modular mitigations such as removing the prefix, rewriting suspicious lines, or repairing completed programs, but none closes the gap to clean-prefix completion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dinh-2023-large-2306-03438]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dinh-2023-large-2306-03438]].
