---
type: concept
title: Dataflow Summarization
slug: dataflow-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: [data-flow summarization, 数据流摘要]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dataflow Summarization** (数据流摘要) — the derivation of function-level summaries that describe which inputs, outputs, parameters, or internal values may participate in dataflow facts.

## Key Points

- LLMDFA reduces interprocedural analysis to function summaries from candidate sources, parameters, and outputs to candidate sinks, arguments, and returns.
- The paper uses few-shot chain-of-thought prompting so the model reasons through assignments, direct uses, and pointer operations step by step.
- For DBZ, this phase reaches `90.95%` precision and `97.57%` recall; for XSS, it reaches `86.52%` precision and `96.25%` recall.
- The paper identifies this phase as a key source of residual errors when functions become large or semantically complex.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-when-2402-10754]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-when-2402-10754]].
