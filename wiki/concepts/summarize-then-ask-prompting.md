---
type: concept
title: Summarize-then-Ask Prompting
slug: summarize-then-ask-prompting
date: 2026-04-20
updated: 2026-04-20
aliases: [SAP, 先总结后提问提示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Summarize-then-Ask Prompting** (先总结后提问提示) — a two-stage prompting strategy in which an LLM first extracts a salient summary from a passage and then generates a query conditioned on that summary plus the original passage.

## Key Points

- [[thakur-2024-leveraging-2311-05800]] defines SAP with two stages: `e_s = LLM(p_s; θ^1, ..., θ^k)` for extractive summary generation and `q_t = LLM(e_s, p_s; θ^1, ..., θ^k)` for target-language query generation.
- The intermediate summary is intended to highlight relevant evidence inside long passages and reduce uninformative or overly extractive queries.
- On XOR-Retrieve, SAP improves downstream `Recall@5kt` over standard prompting for every PaLM 2 size tested, with gains ranging from `+0.6` to `+3.2`.
- The method still depends on few-shot exemplars and does not work well in zero-shot settings for this multilingual task.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[thakur-2024-leveraging-2311-05800]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[thakur-2024-leveraging-2311-05800]].
