---
type: concept
title: Contextualized Exact Lexical Match
slug: contextualized-exact-lexical-match
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contextualized Exact Lexical Match** — an exact-match retrieval scheme that scores only shared query-document tokens, but computes their contribution from contextualized token embeddings rather than heuristic statistics.

## Key Points

- COIL defines token matching as a sum over overlapping terms only, keeping the lexical filter while changing the match signal from `tf`-style heuristics to dot-product similarity.
- For each query token, the model selects the maximum similarity over all identical document-token mentions, so context determines which occurrence contributes most.
- The approach aims to preserve efficiency and interpretability while correcting semantic mismatch among same-surface-form tokens.
- It can be trained end-to-end with a differentiable ranking loss, unlike many hand-engineered lexical scoring functions.
- In the paper, contextualized exact match alone already beats BM25, DeepCT, DocT5Query, and the reproduced dense retriever on passage retrieval MRR.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2021-coil-2104-07186]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2021-coil-2104-07186]].
