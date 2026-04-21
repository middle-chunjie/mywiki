---
type: entity
title: RefChecker
slug: refchecker
date: 2026-04-20
entity_type: tool
aliases: [reference-based fine-grained hallucination checker]
tags: [framework]
---

## Description

RefChecker is the open-sourced claim extraction and entailment checking framework used inside the RAGChecker pipeline. In this paper, Llama3-70B-based RefChecker serves as both the extractor and checker.

## Key Contributions

- Supplies claim extraction for responses and ground-truth answers.
- Supplies entailment checks against retrieved chunks and answer texts.
- Validates the paper's fine-grained metric pipeline with a strong open-source checker setup.

## Related Concepts

- [[claim-extraction]]
- [[claim-verification]]
- [[hallucination]]

## Sources

- [[ru-2024-ragchecker-2408-08067]]
