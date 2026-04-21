---
type: entity
title: AIR Benchmark
slug: air-benchmark
date: 2026-04-20
entity_type: benchmark
aliases:
  - AIR-Bench
  - AIR benchmark
tags: []
---

## Description

AIR Benchmark is an out-of-domain information retrieval benchmark with QA and Long-Doc tracks, used in [[lee-2024-nvembed-2405-17428]] to test generalization beyond [[mteb]].

## Key Contributions

- Provides a closed-book retrieval evaluation over healthcare, law, news, books, arXiv, finance, and other domains.
- Shows that NV-Embed-v2 reaches `52.28` average nDCG@10 on QA and `74.78` average Recall@10 on Long-Doc.

## Related Concepts

- [[dense-retrieval]]
- [[retrieval-augmented-generation]]
- [[contrastive-learning]]

## Sources

- [[lee-2024-nvembed-2405-17428]]
