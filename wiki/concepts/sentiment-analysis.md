---
type: concept
title: Sentiment Analysis
slug: sentiment-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [opinion mining, 情感分析, sentiment classification]
tags: [nlp, sentiment, opinion-mining]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sentiment Analysis** (情感分析) — the NLP task of identifying and classifying the sentiment polarity (positive, negative, neutral) or sentiment intensity expressed in natural language text; ranges from document-level to sentence-level and aspect-level granularity.

## Key Points

- Document- and sentence-level sentiment analysis identifies overall polarity; aspect-based sentiment analysis (ABSA) is a finer-grained subfield that targets polarity for specific aspects within a sentence.
- Deep learning approaches using LSTMs with attention mechanisms (ATAE-LSTM, IAN, MemNet) significantly advanced the field before large pre-trained models; BERT-based methods further pushed state-of-the-art.
- The task has broad applications including recommender systems (sentiment-driven item recommendations) and question answering.
- Key challenge is aspect-conditioned understanding: a sentence like "the food is terrible but the service is great" requires the model to correctly assign opposing polarities to different aspects.
- Standard evaluation metrics are accuracy and macro F1-score over positive/negative/neutral classes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-incorporating]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-incorporating]].
