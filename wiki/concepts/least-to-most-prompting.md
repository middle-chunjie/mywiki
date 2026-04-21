---
type: concept
title: Least-to-Most Prompting
slug: least-to-most-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [least to most prompting, L2M prompting, L2M, 从易到难提示]
tags: [prompting, reasoning, llm, few-shot]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Least-to-Most Prompting** (从易到难提示) — a two-stage few-shot prompting strategy where a complex problem is first decomposed into a sequence of simpler subproblems, then solved sequentially so each step is conditioned on the answers to prior subproblems, enabling in-context generalization to problems harder than the demonstration exemplars.

## Key Points

- Stage 1 uses a decomposition prompt to produce an ordered list of simpler subproblems from the original question; Stage 2 uses a solution prompt that appends each solved subproblem before addressing the next.
- The method is rooted in educational psychology's least-to-most prompting technique, which uses progressive hints to scaffold learners.
- On SCAN length split, `code-davinci-002` with L2M reaches `99.7%` accuracy using only 14 exemplars, while chain-of-thought achieves `16.2%` and neural-symbolic models trained on 15,000+ examples are needed otherwise.
- A single-pass variant merges decomposition and solving into one prompt (applied for GSM8K), trading accuracy for efficiency.
- Performance gains over CoT are largest when test problems require significantly more steps than the demonstration examples.
- Decomposition prompts are not cross-domain transferable: a prompt for math word problems will not teach compositional decomposition for commonsense questions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-leasttomost-2205-10625]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-leasttomost-2205-10625]].
