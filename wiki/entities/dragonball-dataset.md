---
type: entity
title: DragonBall Dataset
slug: dragonball-dataset
date: 2026-04-20
entity_type: dataset
aliases: [DragonBall, Diverse RAG Omni-Benchmark for All scenarios]
tags: [rag, benchmark, multilingual, evaluation]
---

## Description

DragonBall (Diverse RAG Omni-Benchmark for All scenarios) is a multilingual Chinese–English RAG evaluation benchmark released as part of the RAGEval framework, covering finance, law, and medical domains with 6,711 questions generated via a [[schema-based-generation]] pipeline.

## Key Contributions

- Contains 6,711 QRA triples across three domains (finance: 20 corporate scenarios; law: 10 legal scenarios; medical: 19 major categories) in both Chinese and English.
- Features 7 question types: Factual (FQ), Summarization (SQ), Multi-hop Reasoning (MRQ), Information Integration (IIQ), Numerical Comparison (NCQ), Temporal Sequence (TSQ), and Unanswerable (UQ).
- Released under CC-BY-NC license at https://github.com/OpenBMB/RAGEval; documents generated with `chunk_size = 512`, `TopK = 5`, meta-information augmentation.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[schema-based-generation]]
- [[benchmark-evaluation]]
- [[multihop-question-answering]]

## Sources

- [[zhu-2024-rageval-2408-01262]]
