---
type: entity
title: IWSLT'14 De-En
slug: iwslt-14-de-en
date: 2026-04-20
entity_type: dataset
aliases: [IWSLT 2014 De-En, IWSLT14 German-English, iwslt14]
tags: [dataset, machine-translation, benchmark]
---

## Description

IWSLT'14 De-En is a German-to-English machine translation benchmark consisting of approximately 170K sentence-pair training examples drawn from TED talk transcripts. It is a standard small-scale MT evaluation task widely used as a proof-of-concept for retrieval-augmented generation approaches.

## Key Contributions

- Used in TRIME to evaluate generalization of in-batch memory training to sequence-to-sequence tasks.
- TRIME-MT achieves 33.73 BLEU on this dataset, outperforming kNN-MT (33.15) and vanilla Transformer (32.58).
- Short average sentence length (~25 tokens) makes long-term and local memory less relevant; only external memory (BM25-batched target sentences) is applied.

## Related Concepts

- [[knn-machine-translation]]
- [[domain-adaptation]]

## Sources

- [[zhong-2022-training-2205-12674]]
