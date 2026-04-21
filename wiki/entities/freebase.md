---
type: entity
title: Freebase
slug: freebase
date: 2026-04-20
entity_type: knowledge-graph
aliases: [Freebase Knowledge Graph]
tags: []
---

## Description

Freebase is the knowledge graph used by [[ding-2024-enhancing-2402-02175]] for both retrieval and answer reasoning. The paper builds its RR-AP index from the latest Freebase dump and serializes candidate evidence patterns with Freebase labels.

## Key Contributions

- Supplies the entity-relation graph over which EPR performs atomic-pattern retrieval and evidence-pattern instantiation.
- Provides `2,366,590` relation-relation atomic patterns for the FAISS retrieval index used in the paper.

## Related Concepts

- [[knowledge-graph-question-answering]]
- [[atomic-pattern]]
- [[dense-retrieval]]

## Sources

- [[ding-2024-enhancing-2402-02175]]
