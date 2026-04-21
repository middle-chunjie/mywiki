---
type: concept
title: kNN Machine Translation
slug: knn-machine-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [kNN-MT, k-nearest neighbor machine translation, 最近邻机器翻译]
tags: [machine-translation, retrieval, knn, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**kNN Machine Translation** (最近邻机器翻译) — a retrieval-augmented translation method (Khandelwal et al., 2021) that augments a neural encoder-decoder model by interpolating its softmax distribution with a distribution from nearest-neighbor search over a datastore of training-set target token contexts.

## Key Points

- Proposed by Khandelwal et al. (2021) at ICLR as a direct extension of kNN-LM to the sequence-to-sequence setting, using the decoder hidden state as the query key.
- Datastore construction: context-target pairs `((x, y_{<t}), y_t)` for all training translation pairs; the output representation of the final decoder layer is used as the key.
- TRIME-MT (TRIMEMText) adapts TRIME to machine translation by training with BM25-batched target sentences (same-domain similar segments in the same batch) and applying the contrastive in-batch memory objective during training.
- On IWSLT'14 De-En, TRIME-MT achieves 33.73 BLEU vs. 33.15 for kNN-MT and 32.58 for vanilla Transformer, demonstrating that joint training with memory improves over test-time-only augmentation in MT as well.
- Limitation: evaluated only on IWSLT'14 (small dataset, 160K pairs); generalization to larger MT benchmarks is unconfirmed.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2022-training-2205-12674]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2022-training-2205-12674]].
