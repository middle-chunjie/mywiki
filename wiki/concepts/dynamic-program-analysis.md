---
type: concept
title: Dynamic Program Analysis
slug: dynamic-program-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [runtime analysis, 动态程序分析]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Program Analysis** (动态程序分析) — analysis of program properties through actual executions on concrete inputs rather than only through source-level structure.

## Key Points

- The paper frames test cases and their outputs as dynamic evidence of functionality that complements static code representations.
- FuzzPretrain operationalizes dynamic analysis via input-output pairs rather than richer execution traces, keeping the signal compact enough for language-model pre-training.
- Dynamic information matching explicitly predicts whether a code snippet and a test-case bundle correspond to the same runtime behavior.
- The results show dynamic signals are especially useful for defect detection, where subtle source changes can change observed behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code-2309-09980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code-2309-09980]].
