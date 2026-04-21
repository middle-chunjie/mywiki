---
type: concept
title: Long-Context Benchmark
slug: long-context-benchmark
date: 2026-04-20
updated: 2026-04-20
aliases: [long context benchmark, 长上下文基准]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Context Benchmark** (长上下文基准) — a benchmark designed to evaluate models on tasks whose inputs require substantially larger context windows than conventional NLP datasets.

## Key Points

- LOFT operationalizes the idea with `6` task families, `35` datasets, `4` modalities, and contexts up to `1M` tokens.
- The benchmark is constructed so many datasets can be re-instantiated at `32k`, `128k`, and `1M` tokens while keeping the evaluation target fixed.
- Smaller retrieval and RAG corpora are subsets of larger ones, enabling scaling studies rather than one-off long-context snapshots.
- The paper argues that realistic long-context evaluation should stress retrieval, reasoning, and instruction following rather than only synthetic needle-style probes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-can-2406-13121]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-can-2406-13121]].
