---
type: entity
title: MassiveDS
slug: massiveds
date: 2026-04-20
entity_type: dataset
aliases: [Massive DS]
tags: []
---

## Description

MassiveDS is the trillion-token retrieval datastore introduced in [[shao-2024-scaling-2407-12854]]. It is a multi-domain open datastore used to study how inference-time memory scale affects retrieval-based language models.

## Key Contributions

- Aggregates `1,441.2B` tokens across `8` domains, with general web data contributing `1191.7B` tokens.
- Enables monotonic scaling experiments showing strong gains on TriviaQA, Natural Questions, MMLU, and language modeling.
- Serves as the shared corpus behind the paper's efficient retrieval, filtering, and subsampling pipeline.

## Related Concepts

- [[retrieval-based-language-model]]
- [[datastore-scaling]]
- [[dense-retrieval]]

## Sources

- [[shao-2024-scaling-2407-12854]]
