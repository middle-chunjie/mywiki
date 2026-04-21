---
type: concept
title: Multilingual Retrieval
slug: multilingual-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [multi-lingual retrieval, cross-language retrieval]
tags: [retrieval, multilingual]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Retrieval** (多语言检索) — retrieval with a single model across multiple languages, either by searching within each language or by sharing representations across languages.

## Key Points

- The paper introduces mContriever, which extends Contriever to multilingual retrieval by initializing from mBERT and pre-training on `29` languages.
- Training samples are drawn uniformly over languages rather than proportional to corpus size, to avoid domination by high-resource languages.
- On Mr. TyDi, multilingual contrastive pre-training plus MS MARCO fine-tuning reaches average `Recall@100 = 87.0`, substantially above BM25's `74.3`.
- The appendix reports a curse of multilinguality: expanding to more languages can reduce performance, so scale alone is not always beneficial.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[izacard-2022-unsupervised-2112-09118]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[izacard-2022-unsupervised-2112-09118]].
