---
type: entity
title: RocketQA
slug: rocketqa
date: 2026-04-20
entity_type: tool
aliases: [RocketQA]
tags: []
---

## Description

RocketQA is the prior dense retrieval and reranking system that initializes both components in [[ren-2023-rocketqav-2110-07367]] and also supplies retrieval and denoising machinery for data augmentation.

## Key Contributions

- Provides the pretrained dual-encoder retriever and cross-encoder re-ranker used to initialize RocketQAv2.
- Supplies top-`n` retrieval results and denoised confidence signals for hybrid data augmentation.
- Serves as the main direct baseline that RocketQAv2 improves on both retrieval and reranking metrics.

## Related Concepts

- [[dense-passage-retrieval]]
- [[passage-re-ranking]]
- [[hard-negative-mining]]

## Sources

- [[ren-2023-rocketqav-2110-07367]]
