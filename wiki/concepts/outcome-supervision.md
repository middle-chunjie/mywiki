---
type: concept
title: Outcome Supervision
slug: outcome-supervision
date: 2026-04-20
updated: 2026-04-20
aliases: [result supervision, 结果监督]
tags: [llm, reasoning, alignment]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Outcome Supervision** (结果监督) — a supervision scheme that labels a solution by its final result without directly checking the correctness of intermediate reasoning steps.

## Key Points

- The paper's outcome-supervised reward model is trained to predict whether a complete solution is correct or incorrect from its final answer.
- At test time, the ORM uses the score at the final token as the overall score for a candidate solution.
- Final-answer grading can assign false positives to spurious chains of thought that accidentally reach the correct answer.
- Even when large matched synthetic datasets are used, outcome supervision remains weaker than process supervision in best-of-`N` search.
- The paper treats outcome supervision as a harder credit-assignment problem because the reward model must infer where the reasoning first went wrong.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lightman-2023-lets-2305-20050]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lightman-2023-lets-2305-20050]].
