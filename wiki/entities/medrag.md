---
type: entity
title: MedRAG
slug: medrag
date: 2026-04-20
entity_type: tool
aliases: []
tags: [rag, medical, baseline]
---

## Description

MedRAG is a retrieval-augmented generation system for medical question answering introduced alongside the MIRAGE benchmark (Xiong et al., 2024). It uses a single fixed chunk granularity with BM25 retrieval over the MedCorp corpus and serves as the primary RAG baseline for MoG evaluation.

## Key Contributions

- Established a reproducible single-granularity RAG baseline for medical QA.
- Defined the default chunk size used in MoG's second granularity level to enable direct comparison.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[medical-question-answering]]
- [[bm25]]

## Sources

- [[zhong-2024-mixofgranularity-2406-00456]]
