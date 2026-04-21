---
type: concept
title: Execution Feedback
slug: execution-feedback
date: 2026-04-20
updated: 2026-04-20
aliases: [feedback from execution, 执行反馈]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Execution Feedback** (执行反馈) — verification signals obtained by comparing a model's prediction with the behavior of executable reference code and then feeding the mismatch or success signal back into training or revision.

## Key Points

- In CodeI/O++, incorrect first-turn predictions are checked against the reference program and converted into feedback for a second prompt turn.
- The authors use this feedback instead of discarding wrong samples through rejection sampling, preserving more data diversity.
- Successful first-turn predictions receive a simple success signal, while failed predictions trigger a revised response attempt.
- The paper frames execution feedback as a way to improve reasoning data quality without collapsing everything to bare ground-truth answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-codeio-2502-07316]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-codeio-2502-07316]].
