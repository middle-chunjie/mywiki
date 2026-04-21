---
type: concept
title: Execution-Based Decoding
slug: execution-based-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [execution-guided decoding, 基于执行的解码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Execution-Based Decoding** (基于执行的解码) — a decoding or reranking strategy that uses program execution behavior, rather than only token probabilities or lexical similarity, to choose among generated code candidates.

## Key Points

- [[shi-2022-natural-2204-11454]] implements execution-based decoding by comparing sampled programs through their outputs on available test inputs.
- The approach is inference-time only: generated programs are executed after sampling, and the final answer is selected from the candidate pool.
- The paper shows that even one test input per problem can provide a useful semantic signal for reranking on MBPP and Spider.
- For settings where real execution is unavailable, the paper uses a weaker approximation on NL2Bash, illustrating both the utility and limits of execution-grounded selection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-natural-2204-11454]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-natural-2204-11454]].
