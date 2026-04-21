---
type: entity
title: LoCo Benchmark
slug: loco-benchmark
date: 2026-04-20
entity_type: tool
aliases: [LoCo, long context benchmark]
tags: []
---

## Description

LoCo Benchmark is a long-context retrieval benchmark used in [[nussbaum-2025-nomic-2402-01613]] to evaluate embedding models across multiple retrieval datasets and sequence lengths.

## Key Contributions

- Measures long-context retrieval behavior at lengths including `2048`, `4096`, and `8192`.
- Shows [[nomic-embed-text-v1]] outperforming Ada-002 and `text-embedding-3-small` on the reported average.

## Related Concepts

- [[text-embedding]]
- [[retrieval-augmented-generation]]
- [[dynamic-ntk-scaling]]

## Sources

- [[nussbaum-2025-nomic-2402-01613]]
