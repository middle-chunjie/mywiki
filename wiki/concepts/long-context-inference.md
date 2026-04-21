---
type: concept
title: Long-Context Inference
slug: long-context-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [long context processing, 长上下文推理]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Context Inference** (长上下文推理) — running a model on inputs whose document or dialogue history is long enough that memory, latency, and context-window limits become dominant constraints.

## Key Points

- [[li-2023-compressing]] studies long-context inference on documents and conversations rather than on short single-turn QA benchmarks.
- The paper evaluates the method on `408` arXiv documents, `294` BBC News articles, and `470` ShareGPT conversations.
- The central claim is that context compression can recover much of the efficiency lost to quadratic attention cost without changing the target LLM.
- The reported benefits are strongest at mild or moderate compression; aggressive pruning harms reconstruction and answer completeness more sharply.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-compressing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-compressing]].
