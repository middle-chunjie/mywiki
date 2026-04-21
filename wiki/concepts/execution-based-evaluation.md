---
type: concept
title: Execution-Based Evaluation
slug: execution-based-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [Execution-Based Evaluation, 基于执行的评估]
tags: [evaluation, testing, code]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Execution-Based Evaluation** (基于执行的评估) — assessing generated code by running it against tests or runtime checks rather than comparing only its text to a reference.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] uses all-unit-tests-pass as the main similarity oracle for SYNTHESISRTC.
- The paper argues execution is a stronger signal than lexical overlap for code correctness.
- Unit tests are treated as partial but practical proxies for semantic equivalence.
- The approach lets RTC scale to repository code when existing test suites are available.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
