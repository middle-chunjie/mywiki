---
type: concept
title: Cross-Lingual Retrieval
slug: cross-lingual-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-language retrieval, multilingual cross-lingual retrieval]
tags: [retrieval, multilingual]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cross-Lingual Retrieval** (跨语言检索) — retrieval where the query and target documents are in different languages, requiring shared semantic matching beyond lexical overlap.

## Key Points

- The paper evaluates a setting derived from MKQA where non-English queries retrieve documents from English Wikipedia.
- This setting is hard for lexical systems such as BM25 because term overlap can vanish across languages and scripts, for example Arabic to English.
- mContriever fine-tuned only on English MS MARCO still reaches average `R@100 = 65.6`, outperforming CORA at `59.8`.
- The results suggest contrastive multilingual pre-training transfers semantic retrieval ability across languages even without supervised cross-lingual labels.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[izacard-2022-unsupervised-2112-09118]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[izacard-2022-unsupervised-2112-09118]].
