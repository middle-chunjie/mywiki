---
type: concept
title: Code Style Similarity
slug: code-style-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [CSSim, code style similarity, 代码风格相似度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Style Similarity** (代码风格相似度) — an automatic metric that measures how similar two code snippets are in implementation style rather than only in surface token overlap or task semantics.

## Key Points

- The metric is introduced to explain why query-generated exemplar code and corpus code may be semantically aligned yet still stylistically mismatched for retrieval.
- CSSim decomposes style into three components: variable naming, API invocation, and code structure.
- Variable-name and API similarity are computed with IDF-weighted nearest-neighbor edit distance, which softens exact-match requirements.
- Structural similarity is computed with tree-edit distance over the two programs' abstract syntax trees.
- The final style distance is `CSDis = (Dis_Var + Dis_API + TED) / 3`, and similarity is defined as `1 - CSDis`.
- In the paper, CSSim correlates with ReCo's retrieval improvements more consistently than CodeBLEU, ROUGE-L, or BLEU.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-rewriting-2401-04514]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-rewriting-2401-04514]].
