---
type: entity
title: TransE
slug: transe
date: 2026-04-20
entity_type: tool
aliases: [Translating Embeddings for Modeling Multi-relational Data]
tags: []
---

## Description

TransE is a translational knowledge graph embedding method used in [[huang-2023-dual]] to pretrain item and relation embeddings before sequential modeling.

## Key Contributions

- Supplies relation embeddings that SCEL uses to encode complementary and substitute product semantics.
- Enables item-level and category-level relation pretraining with an `L2` translation objective.

## Related Concepts

- [[knowledge-graph-embedding]]
- [[semantics-enhanced-context-embedding-learning]]

## Sources

- [[huang-2023-dual]]
