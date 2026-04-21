---
type: concept
title: Data Mixture
slug: data-mixture
date: 2026-04-20
updated: 2026-04-20
aliases: [training mixture, 数据混合配比]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Mixture** (数据混合配比) — the allocation of training tokens across data sources or domains, such as natural language text, code, math, or instruction data, during model training.

## Key Points

- [[unknown-nd-code]] treats data-mixture design as a primary experimental variable rather than a fixed implementation detail.
- The paper trains matched models with code proportions of `0%`, `25%`, `50%`, `75%`, `90%`, and `100%` over `200B` tokens.
- A moderate mixture with `25%` code and `75%` text gives the best from-scratch natural-language reasoning result in the proportion sweep.
- World-knowledge performance deteriorates as the code share grows, while code performance improves almost linearly with more code.
- The recommended overall recipe uses different mixtures at different stages: balanced code-text pretraining, text-heavy continuation with some code retained, and a cooldown that again includes code.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-code]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-code]].
