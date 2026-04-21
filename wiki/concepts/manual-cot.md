---
type: concept
title: Manual Chain-of-Thought
slug: manual-cot
date: 2026-04-20
updated: 2026-04-20
aliases: [Manual-CoT, manual chain of thought, manual chain-of-thought prompting, 手工思维链]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Manual Chain-of-Thought** (手工思维链) — a few-shot prompting paradigm in which human annotators hand-craft demonstrations consisting of a question, a step-by-step rationale, and an expected answer, which are prepended to the test query to elicit reasoning in a language model.

## Key Points

- Introduced by Wei et al. (2022) as the original few-shot CoT prompting setup; consistently outperforms Zero-Shot-CoT on complex reasoning benchmarks.
- Typical configuration uses `k = 8` hand-crafted demonstrations; demonstration quality is critical — changing the annotator causes up to `28.2%` accuracy disparity on symbolic reasoning tasks.
- The rationale-answer consistency is the most critical component: shuffling answers degrades accuracy from `91.7%` to `17.0%` on MultiArith; shuffling rationales degrades to `43.8%`.
- Manual-CoT often reuses the same demonstrations across multiple related tasks (e.g., 5 of 6 arithmetic datasets share demonstrations), trading task specificity for reduced annotation cost.
- Auto-CoT was proposed to eliminate the manual effort while matching Manual-CoT performance by replacing hand-crafted demonstrations with diversity-sampled, auto-generated ones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-automatic-2210-03493]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-automatic-2210-03493]].
