---
type: entity
title: NomicBERT
slug: nomic-bert
date: 2026-04-20
entity_type: tool
aliases:
  - Nomic BERT
tags: []
---

## Description

NomicBERT is the pretrained backbone family selected in [[morris-2024-contextual-2410-02525]] for the paper's contextual embedding experiments. The authors use a `137M`-parameter variant as the base encoder underlying both stages of the CDE model.

## Key Contributions

- Provides the shared backbone initialization for `M_1` and `M_2` in the contextual encoder.
- Supports the paper's large-scale MTEB experiments with a sub-`250M`-parameter model.

## Related Concepts

- [[document-embedding]]
- [[contextual-document-embedding]]
- [[contrastive-learning]]

## Sources

- [[morris-2024-contextual-2410-02525]]
