---
type: entity
title: ERNIE-2.0
slug: ernie-2-0
date: 2026-04-20
entity_type: tool
aliases: [ERNIE 2.0, ERNIE-2.0 base]
tags: []
---

## Description

ERNIE-2.0 is the BERT-like pretrained backbone used for both the retriever and re-ranker in [[ren-2023-rocketqav-2110-07367]], where the paper follows the base 12-layer configuration.

## Key Contributions

- Supplies the shared pretrained language-model backbone for both dual-encoder retrieval and cross-encoder reranking.
- Lets the paper compare architectural and training gains against a same-backbone DPR-E baseline.

## Related Concepts

- [[dense-passage-retrieval]]
- [[cross-encoder]]
- [[knowledge-distillation]]

## Sources

- [[ren-2023-rocketqav-2110-07367]]
