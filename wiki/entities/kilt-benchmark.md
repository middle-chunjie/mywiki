---
type: entity
title: KILT Benchmark
slug: kilt-benchmark
date: 2026-04-20
entity_type: dataset
aliases: [KILT, Knowledge Intensive Language Tasks, KILT leaderboard]
tags: [benchmark, retrieval, nlp, knowledge-intensive]
---

## Description

KILT (Knowledge Intensive Language Tasks) is a unified benchmark for evaluating knowledge-intensive NLP tasks, introduced by Petroni et al. (2021, NAACL). It provides a shared Wikipedia retrieval corpus and standardized provenance (supporting document) labels across 11 datasets spanning QA, fact verification, slot-filling, entity linking, and dialogue.

## Key Contributions

- Provides a single Wikipedia dump (100-word passages) as the shared retrieval corpus for all tasks, enabling controlled comparison of retrieval strategies across task types.
- Defines KILT-Score metrics that jointly evaluate retrieval provenance accuracy and generation quality, requiring correct document retrieval as a prerequisite for score credit.
- Hosts a public leaderboard used in [[zamani-2024-stochastic]] to compare Stochastic RAG against prior systems including FiD, Re2G, GripRank, and others.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[kilt-score]]
- [[knowledge-intensive-generation]]
- [[dense-retrieval]]

## Sources

- [[zamani-2024-stochastic]]
