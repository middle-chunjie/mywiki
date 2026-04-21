---
type: concept
title: Chain-of-Query
slug: chain-of-query
date: 2026-04-20
updated: 2026-04-20
aliases: [CoQ, query chain]
tags: [reasoning, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain-of-Query** — a globally planned sequence of IR-oriented query-answer pairs that decomposes a complex question into retrievable reasoning steps.

## Key Points

- SearChain defines `CoQ = (q_1, a_1) -> ... -> (q_n, a_n)` and uses it as the basic interface between the LLM and retrieval.
- Each node contains both a query and an answer, plus an implicit status that can mark the query as `[Unsolved Query]` when the model lacks the needed knowledge.
- The paper argues CoQ preserves global reasoning coherence better than methods such as Self-Ask or ReAct that interleave retrieval with only local sub-question generation.
- After retrieval feedback, the LLM regenerates a new CoQ rooted at the corrected or completed node, so CoQ is re-planned across rounds rather than appended greedily.
- In the no-IR setting, CoQ alone improves reasoning depth: SearChain w/o IR exceeds CoT-style baselines and uses more reasoning steps on MuSiQue.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-searchinthechain-2304-14732]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-searchinthechain-2304-14732]].
