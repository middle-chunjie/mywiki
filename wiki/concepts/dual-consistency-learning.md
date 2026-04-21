---
type: concept
title: Dual-Consistency Learning
slug: dual-consistency-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [dual consistency learning]
tags: [retrieval, training]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dual-Consistency Learning** (双一致性学习) — a retrieval training strategy that jointly aligns multiple representation heads by sharing hard negatives and enforcing agreement between their ranking distributions.

## Key Points

- In [[shen-2023-unifier-2205-11194]], it is the central training recipe that couples UnifieR's dense and lexicon retrieval heads instead of optimizing them independently.
- It combines two mechanisms: negative-bridged self-adversarial learning from `N^(den) U N^(lex)` and agreement-based self-regularization via `D_KL(P_den || P_lex)`.
- The method starts from BM25-mined negatives during warmup, then switches to hard negatives mined by the model's own dense and sparse views.
- Its empirical impact is strongest on the dense head: removing self-adversarial and self-regularization lowers dense MRR@10 from `38.8` to `37.9` on MS MARCO Dev.
- The underlying intuition is that dense and lexicon retrieval encode complementary global and local evidence, so consistency constraints can transfer useful supervision across heads.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shen-2023-unifier-2205-11194]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shen-2023-unifier-2205-11194]].
