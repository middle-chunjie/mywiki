---
type: concept
title: Multilingual Question Generation
slug: multilingual-question-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [多语言问题生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Question Generation** (多语言问题生成) — the task of generating well-formed questions in one or more target languages from source passages while preserving semantic relevance to the passage content.

## Key Points

- [[thakur-2024-leveraging-2311-05800]] treats multilingual question generation as the bottleneck for synthesizing retrieval supervision in low-resource languages.
- The paper distinguishes cross-lingual generation (`s != t`) from monolingual generation (`s = t`) and uses different prompt-exemplar construction for the two settings.
- PaLM 2 Small is used as the generator because it offers a favorable cost-speed tradeoff relative to larger models.
- The authors report that few-shot prompting is necessary because zero-shot generation often yields malformed or unparseable outputs.
- Human validation shows high language correctness (`99%` to `100%`) but lower adequacy for some languages, especially when passages are short or noisy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[thakur-2024-leveraging-2311-05800]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[thakur-2024-leveraging-2311-05800]].
