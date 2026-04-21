---
type: entity
title: DeBERTa
slug: deberta
date: 2026-04-20
entity_type: model
aliases:
  - DeBERTa-base
tags: []
---

## Description

DeBERTa is the encoder backbone used in [[samarinas-2024-procis]] for the proactive engagement classifier. The paper applies a DeBERTa-base binary model to decide whether a conversation turn warrants document retrieval.

## Key Contributions

- Provides the proactive classifier that gates when retrieval is triggered.
- Enables evaluation of proactive retrieval pipelines that combine timing prediction with ranked document generation.

## Related Concepts

- [[proactive-retrieval]]
- [[conversational-information-seeking]]
- [[normalized-proactive-discounted-cumulative-gain]]

## Sources

- [[samarinas-2024-procis]]
