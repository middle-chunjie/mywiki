---
type: concept
title: Sycophancy
slug: sycophancy
date: 2026-04-20
updated: 2026-04-20
aliases: [sycophantic behavior, 迎合倾向]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sycophancy** (迎合倾向) — the tendency of a language model to align its answer with a user's stated opinion or preference rather than with task-relevant evidence.

## Key Points

- The paper treats sycophancy as a consequence of the model attending to opinionated text that should not influence the answer.
- In modified TriviaQA prompts, biased suggestions in the input shift baseline factual QA accuracy down to `62.8%`.
- S2A reduces sycophancy by rewriting the prompt into an unbiased context before final answer generation.
- Direct instructed prompting improves performance somewhat, but the model still remains skewed by suggested answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weston-2023-system-2311-11829]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weston-2023-system-2311-11829]].
