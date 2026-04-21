---
type: entity
title: STS Benchmark
slug: sts-benchmark
date: 2026-04-20
entity_type: dataset
aliases: [STS-B, STS Benchmark]
tags: []
---

## Description

STS Benchmark is a semantic textual similarity dataset used throughout the paper for development-time tuning and test reporting. It is one of the central benchmarks used to compare SimCSE against prior sentence embedding methods.

## Key Contributions

- Serves as the main development set for ablations on dropout, hard negatives, pooling, and temperature.
- Provides one of the seven benchmark scores included in the paper's averaged STS metric.
- Highlights the gap between simple SimCSE variants and earlier baselines, such as `82.5` versus `74.2` in the dropout ablation.

## Related Concepts

- [[semantic-textual-similarity]]
- [[sentence-embedding]]
- [[cosine-similarity]]

## Sources

- [[gao-2022-simcse-2104-08821]]
