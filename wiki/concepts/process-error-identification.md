---
type: concept
title: Process Error Identification
slug: process-error-identification
date: 2026-04-20
updated: 2026-04-20
aliases: [process error identification, reasoning error localization, step error detection, 过程错误识别]
tags: [llm, reasoning, evaluation, benchmark]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Process Error Identification** (过程错误识别) — the task of determining the earliest step in a multi-step reasoning solution that contains an error, where error types include mathematical mistakes, logical fallacies, conceptual misapplications, and completeness failures.

## Key Points

- Introduced as the core evaluation task of [[processbench]]; models must output the 0-indexed position of the first erroneous step, or `-1` if all steps are correct.
- Four recognised error types: (1) mathematical errors (wrong calculations/algebra), (2) logical errors (invalid deductions), (3) conceptual errors (misapplication of principles), (4) completeness errors (missing necessary conditions).
- Steps after the first error are treated as ambiguous—a step may be locally valid yet globally wrong if it operates on an incorrect premise—so only the earliest error position matters.
- Evaluation metric: F1 of per-class accuracies on erroneous samples and correct samples, balancing models that are overly critical against those that under-identify errors.
- Empirical finding: the proportion of solutions with process errors in correct-answer solutions rises sharply from 3.5% (GSM8K) to 51.8% (Omni-MATH), demonstrating that outcome-correct models frequently have flawed processes on harder problems.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zheng-2024-processbench-2412-06559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zheng-2024-processbench-2412-06559]].
