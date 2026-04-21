---
type: concept
title: Rubric-Based Evaluation
slug: rubric-based-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [rubric rewards, 评分量表评估]
tags: [evaluation, alignment, agents]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Rubric-Based Evaluation** (评分量表评估) — an evaluation scheme that scores a response against a weighted set of explicit quality criteria rather than a single scalar preference judgment.

## Key Points

- DR Tulu scores answers with question-specific rubric sets `R_x = {(r_{x,k}, w_{x,k})}` and a judge that returns `0`, `0.5`, or `1` for each criterion.
- Rubric weights may be positive or negative, allowing the verifier to reward desired behaviors and penalize failure modes.
- In the paper's implementation, rubric scoring is computed from the final answer only, independent of the hidden reasoning trace.
- The authors compare general rubrics, closed-book rubrics, search-based rubrics, and evolving rubrics, and find that search-grounded variants are more concrete and factual.
- Static rubric design is insufficient for long-form deep research because new evidence and new bad behaviors appear as the policy changes during RL.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2025-dr-2511-19399]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2025-dr-2511-19399]].
