---
type: concept
title: Multiple-Choice Question
slug: multiple-choice-question
date: 2026-04-20
updated: 2026-04-20
aliases: [MCQ, multiple choice question, 选择题]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multiple-Choice Question** (选择题) — an assessment item that presents one correct answer together with a set of candidate distractor options.

## Key Points

- MMLU-Pro treats the answer format itself as part of benchmark design by expanding the original `4` choices to `10`, making shortcut guessing materially harder.
- The paper uses GPT-4-Turbo to augment distractors so the added options are plausible rather than obviously irrelevant.
- Human experts remove items that are fundamentally unsuitable for the multiple-choice format, including proof questions, image-dependent questions, and questions lacking enough textual information.
- After review, `83%` of questions still retain all `10` options and the benchmark averages `9.47` options per question.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-mmlupro-2406-01574]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-mmlupro-2406-01574]].
