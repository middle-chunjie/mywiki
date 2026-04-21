---
type: entity
title: uniCOIL
slug: unicoil
date: 2026-04-20
entity_type: tool
aliases: [uniCOIL retriever]
tags: []
---

## Description

uniCOIL is the sparse retriever used in [[lin-2023-how-2302-07452]] as the first teacher in progressive label augmentation. It provides an efficient lexical-style supervision signal distinct from dense and late-interaction teachers.

## Key Contributions

- Serves as the easiest teacher in the supervision trajectory `uniCOIL -> Contriever -> GTR-XXL -> ColBERTv2 -> SPLADE++`.
- Helps diversify relevance labels beyond what a single dense or cross-encoder teacher can provide.

## Related Concepts

- [[dense-retrieval]]
- [[data-augmentation]]
- [[curriculum-learning]]

## Sources

- [[lin-2023-how-2302-07452]]
