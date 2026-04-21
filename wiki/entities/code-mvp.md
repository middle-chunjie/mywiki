---
type: entity
title: CODE-MVP
slug: code-mvp
date: 2026-04-20
entity_type: tool
aliases: [Code-MVP]
tags: [model, pretraining]
---

## Description

CODE-MVP is the pretrained code representation model proposed in [[wang-2022-codemvp-2205-02029]]. It combines multiple compiler-derived code views with contrastive pretraining, masked modeling, and type inference.

## Key Contributions

- Introduces a unified pretraining framework over `NL`, `PL`, `AST`, `CFG`, and `PT` views.
- Adds fine-grained type inference and multi-view MLM on top of the contrastive objective.
- Sets the best reported numbers in the paper on retrieval, similarity, and defect detection benchmarks.

## Related Concepts

- [[code-representation-learning]]
- [[multi-view-learning]]
- [[contrastive-learning]]

## Sources

- [[wang-2022-codemvp-2205-02029]]
