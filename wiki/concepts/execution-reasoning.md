---
type: concept
title: Execution Reasoning
slug: execution-reasoning
date: 2026-04-20
updated: 2026-04-20
aliases: [program execution reasoning, 执行推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Execution Reasoning** (执行推理) — reasoning about how a program transforms state during execution and what inputs or outputs are consistent with that behavior.

## Key Points

- [[ding-2024-semcoder-2406-01006]] evaluates execution reasoning with CRUXEval and LiveCodeBench-CodeExecution, focusing on both output prediction and input prediction.
- The model is trained to explain state transitions in natural language instead of memorizing only final outputs or literal traces.
- Semantic-heavy supervision improves execution reasoning much more than generic chain-of-thought prompting or concrete trace formats.
- Compared with DeepSeekCoder-6.7B, SEMCODER improves CRUXEval-I by `+23.0` points and CRUXEval-O by `+23.9` points.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-semcoder-2406-01006]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-semcoder-2406-01006]].
