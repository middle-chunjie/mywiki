---
type: concept
title: Counterfactual Robustness
slug: counterfactual-robustness
date: 2026-04-20
updated: 2026-04-20
aliases: [robustness to false retrieved evidence]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Counterfactual Robustness** — the ability of a retrieval-augmented model to detect and correct false retrieved facts even when the model already has the correct knowledge internally.

## Key Points

- RGB creates this testbed from questions the model can already answer, then manually edits retrieved documents so that answer-bearing spans become factually wrong.
- The evaluation includes both error detection and error correction after the model is warned that retrieved documents may contain factual errors.
- Performance collapses under false context: ChatGPT-en falls from `89%` direct-answer accuracy to `9%` when counterfactual documents are added.
- The paper concludes that current LLMs over-trust retrieved evidence and lack strong safeguards against misinformation in practical RAG pipelines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2023-benchmarking-2309-01431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2023-benchmarking-2309-01431]].
