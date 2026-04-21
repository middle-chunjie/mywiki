---
type: entity
title: T-REx Dataset
slug: t-rex-dataset
date: 2026-04-20
entity_type: dataset
aliases: [T-REx, T-Rex, T-REx slot-filling]
tags: [dataset, slot-filling, relation-extraction, knowledge-base]
---

## Description

T-REx is a large-scale dataset that aligns natural language text with Wikidata knowledge base triples, introduced by Elsahar et al. (LREC 2018). It is used as a slot-filling benchmark in the KILT framework, where models must retrieve a supporting Wikipedia passage and generate the correct entity to fill a relation triple slot.

## Key Contributions

- Provides large-scale alignment of Wikipedia sentences with Wikidata subject-relation-object triples for slot-filling and relation extraction.
- Included in the [[kilt-benchmark]] as one of two slot-filling datasets (alongside zsRE), evaluated with KILT-AC (Accuracy) metric.
- In [[zamani-2024-stochastic]], Stochastic RAG improves FiD-Light XL from KILT-AC 76.3 to 78.3 on T-REx (KILT test set).

## Related Concepts

- [[retrieval-augmented-generation]]
- [[kilt-score]]

## Sources

- [[zamani-2024-stochastic]]
