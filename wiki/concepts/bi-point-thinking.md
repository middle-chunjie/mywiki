---
type: concept
title: Bi-point Thinking
slug: bi-point-thinking
date: 2026-04-20
updated: 2026-04-20
aliases: [solution-comment alternation, dual-point thinking]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bi-point Thinking** — a reasoning pattern that alternates between proposing solutions and generating review comments so that candidate outputs can be iteratively corrected against the original constraints.

## Key Points

- The paper introduces bi-point thinking to address the fact that one-shot generated solutions may fail some real-world constraints.
- Solution nodes hold draft plans, while comment nodes record deficiencies found during review.
- New solutions are conditioned on the prior solution, the review comment, and retrieved domain knowledge.
- This alternation is the paper's main mechanism for gradually increasing completeness and reliability.
- Removing bi-point thinking drops overall performance from `66.2/64.1` to `62.9/61.5`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-deepsolution-2502-20730]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-deepsolution-2502-20730]].
