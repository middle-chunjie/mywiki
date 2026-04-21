---
type: entity
title: RoBERTa
slug: roberta
date: 2026-04-20
entity_type: model
aliases: [Roberta, Robustly Optimized BERT Pretraining Approach]
tags: []
---

## Description

RoBERTa is a pre-trained Transformer encoder used by SimCSE as one of the initialization backbones. In this paper it yields the strongest STS results, especially in the supervised large-model setting.

## Key Contributions

- Serves as a stronger encoder initialization than BERT for SimCSE in the reported experiments.
- Supports `76.57` average STS for unsupervised RoBERTa-base and `83.76` for supervised RoBERTa-large.
- Demonstrates that the SimCSE objective transfers across multiple pre-trained encoder families.

## Related Concepts

- [[sentence-embedding]]
- [[contrastive-learning]]
- [[semantic-textual-similarity]]

## Sources

- [[gao-2022-simcse-2104-08821]]
