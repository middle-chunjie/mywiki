---
type: concept
title: Condition Mask Verification
slug: condition-mask-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [CMV, conditional mask verification, 条件掩码验证]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Condition Mask Verification** (条件掩码验证) — an arithmetic-task verification method that masks one original condition, usually a numeric value, and tests whether the model can recover it after conditioning on a candidate conclusion.

## Key Points

- [[weng-2023-large-2212-09561]] applies CMV to arithmetic reasoning tasks because masked numbers provide explicit verification targets.
- The paper uses regex-based extraction to identify condition values and replaces each selected value with `X` before asking the model to solve for it.
- Verification scores are computed by matching the recovered value against the original masked condition across repeated samples.
- The paper reports that multi-condition masking performs better than single-condition masking and that CMV generally outperforms TFV on arithmetic tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weng-2023-large-2212-09561]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weng-2023-large-2212-09561]].
