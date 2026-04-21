---
type: entity
title: infini-gram
slug: infini-gram
date: 2026-04-20
entity_type: tool
aliases: [infinigram]
tags: []
---

## Description

Infini-gram is the suffix-array-based engine introduced by the paper to serve arbitrary-`n` and `∞`-gram queries over trillion-token corpora with low latency. It stores tokenized text and suffix-array pointers on disk and supports counting, probability, distribution, and document-retrieval queries.

## Key Contributions

- Replaces infeasible explicit `n`-gram count tables with an on-disk suffix-array index.
- Supports query types including `Count`, `NgramProb`, `NgramDist`, `InfinigramProb`, `InfinigramDist`, and document retrieval.
- Makes large-scale nonparametric language modeling practical enough to complement neural LMs in evaluation.

## Related Concepts

- [[suffix-array]]
- [[n-gram-language-model]]
- [[nonparametric-language-model]]

## Sources

- [[liu-2024-infinigram-2401-17377]]
