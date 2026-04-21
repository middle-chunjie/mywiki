---
type: concept
title: Aspect-based Sentiment Analysis
slug: aspect-based-sentiment-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [ABSA, aspect-level sentiment analysis, 方面级情感分析]
tags: [sentiment, nlp, opinion-mining]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Aspect-based Sentiment Analysis** (方面级情感分析) — a fine-grained NLP task that predicts sentiment polarity (positive, negative, neutral) towards a specific aspect term within a sentence, as opposed to document-level sentiment classification.

## Key Points

- Given a sentence-aspect pair `(S, A)`, the model must identify the sentiment polarity of `S` conditioned on the specific aspect `A`; a single sentence can carry different polarities for different aspects (e.g., "positive" for *"DDR5"* and "negative" for *"system memory"* in the same review).
- Early approaches used rule-based dependency parsing or attention-based LSTMs (ATAE-LSTM, IAN, MemNet); a major performance leap came with BERT fine-tuning and graph-based syntactic methods (RGAT-BERT, T-GCN).
- A core challenge is that vanilla BERT encodes global sentence semantics without adequately conditioning on the given aspect, causing failures in multi-aspect sentences.
- DR-BERT addresses this by inserting a Dynamic Re-weighting Adapter that iteratively selects aspect-relevant words (up to `T = 7` steps), mimicking human sequential semantic comprehension.
- Standard benchmarks include SemEval-2014 Restaurant, SemEval-2014 Laptop, and Twitter datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-incorporating]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-incorporating]].
