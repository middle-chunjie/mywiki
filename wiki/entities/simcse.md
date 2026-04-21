---
type: entity
title: SimCSE
slug: simcse
date: 2026-04-20
entity_type: tool
aliases: [Simple Contrastive Learning of Sentence Embeddings]
tags: []
---

## Description

SimCSE is the sentence embedding framework introduced in this paper, with both unsupervised and supervised variants built on top of pre-trained encoders. The authors also release code and pre-trained models for the method.

## Key Contributions

- Shows that standard dropout is sufficient to create effective positive pairs for unsupervised sentence contrastive learning.
- Reuses NLI entailment and contradiction labels to build a strong supervised sentence embedding objective.
- Sets new STS results across BERT and RoBERTa backbones while motivating analysis through alignment and uniformity.

## Related Concepts

- [[sentence-embedding]]
- [[contrastive-learning]]
- [[uniformity]]

## Sources

- [[gao-2022-simcse-2104-08821]]
