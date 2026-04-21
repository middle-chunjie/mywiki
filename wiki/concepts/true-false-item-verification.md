---
type: concept
title: True-False Item Verification
slug: true-false-item-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [TFV, true false verification, 真假条目验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**True-False Item Verification** (真假条目验证) — a general verification strategy that asks whether the original conditions plus a candidate conclusion are jointly correct, and scores the candidate by the number of affirmative consistency judgments.

## Key Points

- [[weng-2023-large-2212-09561]] uses TFV for commonsense and logical reasoning tasks where exact masked-value recovery is less natural than in arithmetic.
- The prompt appends a question such as "Do it is correct (True or False)?" after the condition set that includes the candidate conclusion.
- Verification scores are obtained by counting how many repeated backward checks return `True`.
- The paper treats TFV as more broadly applicable than CMV, but less targeted because it does not force the model to recover a precise masked condition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weng-2023-large-2212-09561]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weng-2023-large-2212-09561]].
