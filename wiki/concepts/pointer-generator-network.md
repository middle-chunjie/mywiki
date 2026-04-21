---
type: concept
title: Pointer-Generator Network
slug: pointer-generator-network
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-source pointer-generator, pointer generator]
tags: [generation, copying]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pointer-Generator Network** (指针生成网络) — a generation mechanism that mixes vocabulary generation with copying probabilities over source inputs.

## Key Points

- [[guo-2022-modeling]] extends the standard pointer-generator idea into a multi-source module that can copy from both source-code tokens and AST nodes.
- The final token probability is `p_s(w) = lambda_v p_v(w) + lambda_c p_c(w) + lambda_n p_n(w)`, where the mixing weights come from a softmax over decoder and context vectors.
- Copy distributions are computed from additional multi-head attention over encoded code and AST representations, and repeated source tokens are merged by summing attention mass.
- Replacing the proposed module with a prior copying mechanism lowers Java performance from `49.19/32.27/59.59` to `48.59/31.82/58.73`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2022-modeling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2022-modeling]].
