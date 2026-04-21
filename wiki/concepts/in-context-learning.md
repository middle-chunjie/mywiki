---
type: concept
title: In-Context Learning
slug: in-context-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [ICL, in context learning, 上下文学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**In-Context Learning** (上下文学习) — a paradigm in which a language model infers how to solve a task from examples placed in the prompt rather than from parameter updates.

## Key Points

- This paper studies few-shot in-context learning under a realistic workload with many test samples instead of one isolated query.
- The main experiments use `12` manually selected demonstrations per prompt, each with Chain-of-Thought answers.
- Batch prompting preserves the in-context learning setup while changing only how demonstrations and test items are arranged in the prompt.
- The analysis argues that repeated demonstration tokens dominate cost, making in-context learning a natural setting for batched inference.
- The paper validates this design across commonsense QA, arithmetic reasoning, and NLI/NLU tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2023-batch-2301-08721]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2023-batch-2301-08721]].
