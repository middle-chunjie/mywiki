---
type: concept
title: Query-Code Alignment
slug: query-code-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [code-query alignment, 查询-代码对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query-Code Alignment** (查询-代码对齐) — the property that a natural-language query and its semantically matching code snippet occupy nearby positions in a shared embedding space.

## Key Points

- CoCoSoDa explicitly optimizes alignment through inter-modal contrastive loss between paired queries and code snippets.
- The paper treats good retrieval as requiring alignment at the sequence-functionality level rather than only token-level overlap.
- Quantitative analysis uses `l_align` together with `l_uniform` to explain why CoCoSoDa outperforms stronger baselines.
- Case studies show better alignment helps the model retrieve code that matches the intended operation instead of merely sharing surface words.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-cocosoda-2204-03293]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-cocosoda-2204-03293]].
