---
type: concept
title: Semantic Equivalence
slug: semantic-equivalence
date: 2026-04-20
updated: 2026-04-20
aliases: [Semantic Equivalence, 语义等价性]
tags: [semantics, evaluation, code]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Equivalence** (语义等价性) — the condition that two artifacts may differ in surface form but preserve the same meaning or behavior under the task's intended interpretation.

## Key Points

- [[allamanis-2024-unsupervised-2402-08699]] defines RTC in terms of recovering a semantically equivalent `x_hat` from `x`.
- For code synthesis, the paper approximates semantic equivalence with execution against unit tests.
- For code editing, the paper falls back to exact match even though it acknowledges that lexical identity is stricter than true semantic equivalence.
- The paper argues that metrics based only on token overlap can miss semantic correctness.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[allamanis-2024-unsupervised-2402-08699]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[allamanis-2024-unsupervised-2402-08699]].
