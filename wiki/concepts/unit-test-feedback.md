---
type: concept
title: Unit-Test Feedback
slug: unit-test-feedback
date: 2026-04-20
updated: 2026-04-20
aliases: [ut feedback, 单元测试反馈]
tags: [evaluation, debugging, code]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Unit-Test Feedback** (单元测试反馈) — debugging feedback derived from passed or failed test executions and injected back into the model prompt.

## Key Points

- [[chen-2023-teaching-2304-05128]] uses unit-test results directly in TransCoder and MBPP to tell the model how its code failed.
- This feedback is richer than a bare correct/incorrect label because it exposes runtime errors or output mismatches.
- With unit-test feedback plus explanation, Codex reaches `92.5` on TransCoder and `69.8` on MBPP.
- The paper's ablations show that removing execution weakens performance substantially, confirming unit-test feedback is a major source of signal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-teaching-2304-05128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-teaching-2304-05128]].
