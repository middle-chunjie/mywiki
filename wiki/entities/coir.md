---
type: entity
title: CoIR
slug: coir
date: 2026-04-20
entity_type: benchmark
aliases: [CoIR benchmark]
tags: []
---

## Description

CoIR is a benchmark for code information retrieval, used in [[wang-2025-jinarerankerv-2509-25085]] to test whether the reranker transfers to programming-language search. It is the paper's main specialized-domain benchmark outside general text retrieval.

## Key Contributions

- Provides the code-retrieval evaluation where jina-reranker-v3 reports `70.64`.
- Shows that the model's listwise reranking approach is not restricted to natural-language benchmarks such as BEIR or MIRACL.

## Related Concepts

- [[document-reranking]]
- [[dense-retrieval]]

## Sources

- [[wang-2025-jinarerankerv-2509-25085]]
