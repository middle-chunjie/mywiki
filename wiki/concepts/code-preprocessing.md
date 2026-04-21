---
type: concept
title: Code Preprocessing
slug: code-preprocessing
date: 2026-04-20
updated: 2026-04-20
aliases: [source-code preprocessing, code normalization, 代码预处理]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Preprocessing** (代码预处理) — the set of transformations applied to raw source-code tokens before they are used for model training or evaluation.

## Key Points

- The paper studies four common operations: replacing literals, identifier splitting, punctuation filtering, and lowercasing.
- It represents preprocessing choices with the bitwise notation `P_RSFL`, which yields `16` possible combinations.
- Different preprocessing combinations change BLEU-DC substantially, with reported swings from roughly `-18%` to `+25%`.
- No single combination dominates every model, but `P_1101` is consistently strong across the evaluated systems.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2022-evaluation-2107-07112]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2022-evaluation-2107-07112]].
