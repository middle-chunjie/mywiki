---
type: entity
title: MIRAGE Benchmark
slug: mirage-benchmark
date: 2026-04-20
entity_type: tool
aliases:
  - MIRAGE
tags: [benchmark, medical-qa, rag]
---

## Description

MIRAGE (Medical Information Retrieval-Augmented Generation Evaluation) is a benchmark introduced by Xiong et al. (2024) that standardizes evaluation of RAG systems on five medical QA datasets: MMLU-Med, MedQA-US, MedMCQA, PubMedQA, and BioASQ-Y/N, paired with the MedCorp retrieval corpus.

## Key Contributions

- Provides a unified evaluation protocol for RAG in the medical domain with five diverse QA datasets.
- Removes ground-truth supporting contexts during testing to prevent knowledge leakage.
- Establishes MedRAG (single-granularity BM25 retrieval) as the baseline system.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[medical-question-answering]]
- [[mix-of-granularity]]

## Sources

- [[zhong-2024-mixofgranularity-2406-00456]]
