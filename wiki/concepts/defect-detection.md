---
type: concept
title: Defect Detection
slug: defect-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [bug detection, 缺陷检测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Defect Detection** (缺陷检测) — the task of predicting whether a code fragment contains a bug, vulnerability, or faulty implementation behavior.

## Key Points

- The paper evaluates defect detection on Devign to test whether execution-aware pre-training transfers beyond retrieval tasks.
- FuzzCodeBERT improves defect-detection accuracy to `64.1%`, above the `63.5%` code-only MLM baseline and `62.1%` original CodeBERT baseline.
- Dynamic information is especially relevant because subtle code edits can produce different runtime outputs even when syntax looks similar.
- Ablations show that removing dynamic information matching or distillation harms downstream defect detection, supporting the execution-based motivation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code-2309-09980]]
- [[wang-2022-codemvp-2205-02029]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code-2309-09980]].
