---
type: concept
title: Cross-domain Sentiment Classification
slug: cross-domain-sentiment-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [CDSC, cross-domain sentiment analysis, 跨域情感分类]
tags: [sentiment-analysis, domain-adaptation, nlp, transfer-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-domain Sentiment Classification** (跨域情感分类) — a transfer learning task in NLP where a sentiment classifier trained on labeled data from a source domain (e.g., DVD reviews) is adapted to predict sentiment polarity in an unlabeled target domain (e.g., Book reviews), without requiring target-domain labeled data.

## Key Points

- Domain shift is the core challenge: vocabulary and writing style differ across domains, causing performance drops when applying source-trained classifiers directly to the target.
- Early methods relied on manually curated pivot features (structural correspondence learning, POS-tag ensembles); deep learning methods replaced hand-crafted features with learned domain-invariant representations.
- Adversarial training (domain discriminators) is the dominant paradigm for forcing domain-invariant feature extraction.
- GAST (Zhang et al., 2022) shows that syntactic dependency structures are themselves domain-invariant and can serve as transferable features alongside word sequences, addressing a gap in prior sequence-only methods.
- Semi-supervised entropy minimization on unlabeled data from both domains further reduces sentiment discrepancy across domains.
- BERT-based models (BERT-DAAT, BERT-GAST) achieve state-of-the-art by combining large pre-trained language model representations with domain-adversarial fine-tuning.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-graph-2205-08772]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-graph-2205-08772]].
