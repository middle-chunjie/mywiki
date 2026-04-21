---
type: concept
title: Small Language Model
slug: small-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [SLM, 小语言模型]
tags: [language-model, efficiency, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Small Language Model** (小语言模型) — a comparatively low-parameter language model optimized for lower inference cost and lighter hardware requirements than frontier large language models.

## Key Points

- The paper explicitly benchmarks generators from `70M` to `1.3B` parameters to test whether cheap zero-shot generation is still useful for IR.
- Small models are used not only for question generation but also, in the case of GPT-Neo 125M, as probabilistic sources for estimating document normalized information.
- The authors argue that stronger filtering can compensate for weaker generators, allowing small models to produce effective silver-standard datasets.
- Random sampling is especially attractive for small models when the goal is throughput, but the paper also studies deterministic decoding because small models can struggle with naive sampling.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
