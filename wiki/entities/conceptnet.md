---
type: entity
title: ConceptNet
slug: conceptnet
date: 2026-04-20
entity_type: tool
aliases: [ConceptNet 5.5]
tags: []
---

## Description

ConceptNet is the external commonsense knowledge graph used in [[liu-2023-enhancing]] for concept matching, concept pretraining, and knowledge-aware document and label encoding. The paper prunes it to matched concepts and their first-order neighbors before downstream modeling.

## Key Contributions

- Supplies the concept nodes and relations used to enrich token and label representations.
- Enables the shared-concept signal that defines the knowledge-driven contrastive loss.

## Related Concepts

- [[knowledge-graph]]
- [[concept-recognition]]
- [[transe]]

## Sources

- [[liu-2023-enhancing]]
