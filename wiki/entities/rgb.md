---
type: entity
title: RGB
slug: rgb
date: 2026-04-20
entity_type: benchmark
aliases: [RGB benchmark, Retrieval-Augmented Generation Benchmark]
tags: []
---

## Description

RGB is a benchmark for evaluating robustness in [[retrieval-augmented-generation]], and [[wang-2024-astute-2410-07176]] uses its English noise-robustness subset as a worst-case setting where all retrieved documents are negative.

## Key Contributions

- Provides a controlled stress test for RAG under fully unhelpful retrieval.
- Shows that Astute RAG is the only compared RAG method that approaches no-RAG performance in the worst-case setting.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[noise-robustness]]
- [[trustworthiness]]

## Sources

- [[wang-2024-astute-2410-07176]]
