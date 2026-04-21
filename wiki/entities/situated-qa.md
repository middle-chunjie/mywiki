---
type: entity
title: SituatedQA
slug: situated-qa
date: 2026-04-20
entity_type: dataset
aliases: [Situated QA, SituatedQA dataset]
tags: [benchmark, qa, temporal-reasoning]
---

## Description

SituatedQA is an open-retrieval QA benchmark (Zhang & Choi, 2021) that requires models to answer questions given extra-linguistic contexts — specifically temporal or geographical contexts. The test set used in [[zheng-2024-take-2310-06117]] contains 2,901 examples.

## Key Contributions

- Evaluates LLM robustness to contextual grounding; PaLM-2L achieves `54.3%` and GPT-4 achieves `63.2%` baseline accuracy.
- Used in [[zheng-2024-take-2310-06117]] to show a moderate improvement with [[step-back-prompting]] + RAG (`54.3%` → `61%`), slightly below GPT-4.

## Related Concepts

- [[temporal-reasoning]]
- [[retrieval-augmented-generation]]
- [[step-back-prompting]]

## Sources

- [[zheng-2024-take-2310-06117]]
