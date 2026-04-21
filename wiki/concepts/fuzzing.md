---
type: concept
title: Fuzzing
slug: fuzzing
date: 2026-04-20
updated: 2026-04-20
aliases: [fuzz testing, 模糊测试]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Fuzzing** (模糊测试) — an automated dynamic testing method that generates diverse inputs to execute a program, explore new paths, and expose failures or behavioral differences.

## Key Points

- The paper uses a customized fuzzing pipeline to synthesize test cases for `1.2M` CodeNet programs across C/C++, Python, and Java.
- Fuzzed inputs are paired with observed outputs, turning execution behavior into a pre-training signal rather than using fuzzing only for bug finding.
- AFL++ is the concrete fuzzing engine used to generate inputs that cover logic paths as broadly as possible.
- Fuzzing is required only during pre-training; downstream inference consumes code alone after dynamic information has been distilled.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code-2309-09980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code-2309-09980]].
