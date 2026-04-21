---
type: entity
title: FollowIR-7B
slug: followir-7b
date: 2026-04-20
entity_type: tool
aliases: [FollowIR 7B]
tags: []
---

## Description

FollowIR-7B is the instruction-following retrieval model introduced in [[weller-2024-followir-2403-15246]], obtained by fine-tuning Mistral-7B-Instruct-v0.2 on long TREC-style retrieval instructions and synthetic relevance examples.

## Key Contributions

- Achieves the best average instruction-following score in the paper with `p-MRR = +13.6`.
- Improves over the Mistral-7B-Instruct-v0.2 baseline on both standard retrieval quality and pairwise instruction-following evaluation.
- Demonstrates that long-instruction supervision can teach IR models to change rankings in response to altered relevance definitions.

## Related Concepts

- [[instruction-following]]
- [[instruction-tuning]]
- [[reranking]]

## Sources

- [[weller-2024-followir-2403-15246]]
