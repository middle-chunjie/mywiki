---
type: entity
title: FiD-Light
slug: fid-light
date: 2026-04-20
entity_type: model
aliases: [FiD-Light, Fusion-in-Decoder Light, FiD Light]
tags: [rag, retrieval, generation, kilt]
---

## Description

FiD-Light is a retrieval-augmented generation model introduced by Hofstätter et al. (SIGIR 2023) as an efficient extension of Fusion-in-Decoder. It augments the standard FiD architecture with a mechanism to generate relevant document identifiers alongside the output text, enabling inference-time re-ranking of retrieved passages. It was the state-of-the-art model on 6 of 7 KILT benchmark datasets as of February 2024, and serves as the backbone model in [[zamani-2024-stochastic]].

## Key Contributions

- Extends [[fusion-in-decoder]] with a lightweight re-ranking signal: the model simultaneously generates document identifiers of relevant documents, which are used at inference to re-rank the input result list.
- Available in T5-Base (220M parameters, `k=64` passages) and T5-XL (3B parameters, `k=8` passages) variants, demonstrating strong performance across model scales.
- Provides the state-of-the-art baseline on the KILT leaderboard against which [[zamani-2024-stochastic]] measures improvements.

## Related Concepts

- [[fusion-in-decoder]]
- [[retrieval-augmented-generation]]
- [[end-to-end-rag-optimization]]
- [[dense-retrieval]]

## Sources

- [[zamani-2024-stochastic]]
