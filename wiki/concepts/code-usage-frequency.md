---
type: concept
title: Code Usage Frequency
slug: code-usage-frequency
date: 2026-04-20
updated: 2026-04-20
aliases: [code execution frequency]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Usage Frequency** (代码使用频率) — a measure of how many times a model executes code while producing a solution, used to analyze the relationship between executable computation and reasoning performance.

## Key Points

- The paper introduces Code Usage Frequency to distinguish prompt settings that forbid code, allow one code block, or permit unrestricted iterative code use.
- Prompt 1 yields almost no code use, Prompt 2 yields roughly one execution, and the Basic Prompt uses code multiple times across a single solution trace.
- Higher Code Usage Frequency is positively correlated with higher accuracy on MATH across prompt types, subjects, and difficulty levels.
- The correlation is strongest on harder problems, supporting the claim that repeated execution and repair are particularly important for complex mathematical reasoning.
- CSV increases Code Usage Frequency further because it adds explicit verification and possible correction rounds after the initial solution.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-solving-2603-03507]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-solving-2603-03507]].
