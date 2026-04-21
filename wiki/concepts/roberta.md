---
type: concept
title: RoBERTa
slug: roberta
date: 2026-04-20
updated: 2026-04-20
aliases: [Robustly Optimized BERT Pretraining Approach]
tags: [nlp, pretrained-language-model, bert, transformer]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**RoBERTa** — a retraining of BERT (Liu et al., 2019) that achieves stronger performance by using larger batch sizes, higher learning rates, longer training, more data, and dynamic masking, while removing the next-sentence prediction objective.

## Key Points

- Key modifications over BERT: (1) removes next-sentence prediction; (2) uses dynamic (per-epoch) masking instead of static; (3) trains on larger datasets (160 GB); (4) uses larger batches (8K) and longer sequences.
- Outperforms BERT on most downstream NLP benchmarks; adopted as a standard encoder backbone for English NLP tasks in 2019–2022.
- In the CWI setting, RoBERTa produces a `768`-dimensional pooled output `F_c` representing the full sentence context for the target word prediction.
- XLM-RoBERTa extends RoBERTa to 100 languages using a multilingual CommonCrawl corpus, preserving the same architecture but replacing the monolingual training setup.
- Despite its age, RoBERTa remains a competitive encoder for sentence-level tasks in English; domain-adapted fine-tuning continues to yield strong results on specialized domains.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
