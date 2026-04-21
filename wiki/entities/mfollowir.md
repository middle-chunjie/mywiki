---
type: entity
title: mFollowIR
slug: mfollowir
date: 2026-04-20
entity_type: benchmark
aliases: [Multilingual FollowIR]
tags: []
---

## Description

mFollowIR is a multilingual benchmark for instruction-following retrieval used in [[weller-2025-rank-2502-18418]] to test whether Rank1 transfers beyond English training data. It evaluates both cross-lingual and multilingual reranking behavior.

## Key Contributions

- Shows that English-only Rank1 training still yields strong multilingual gains, reaching up to `0.610` average nDCG@20.
- Measures both retrieval quality and instruction adherence through metrics such as `p-MRR`.

## Related Concepts

- [[instruction-following]]
- [[information-retrieval]]
- [[reranking]]

## Sources

- [[weller-2025-rank-2502-18418]]
