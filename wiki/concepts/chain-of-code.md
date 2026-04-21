---
type: concept
title: Chain of Code
slug: chain-of-code
date: 2026-04-20
updated: 2026-04-20
aliases: [CoC, chain of code]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain of Code** — a reasoning framework in which a language model writes code-like intermediate steps and execution interleaves a real interpreter with language-model-based simulation of non-executable lines.

## Key Points

- The paper defines CoC as a two-stage pipeline of code generation followed by hybrid execution.
- Executable lines are run by Python, while semantic or undefined operations are handed to an LMulator that predicts the next program state.
- This lets one reasoning trace mix exact computation with flexible semantic judgments instead of forcing everything into natural language or runnable code alone.
- The method is evaluated primarily on BIG-Bench Hard and GSM8K, where it improves over direct prompting and Chain-of-Thought.
- The authors argue that the same mechanism can extend beyond benchmark QA to robotics tasks involving perception and control APIs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-chain-2312-04474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-chain-2312-04474]].
