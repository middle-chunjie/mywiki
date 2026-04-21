---
type: entity
title: docT5query
slug: doc-t5query
date: 2026-04-20
entity_type: model
aliases:
  - doc2query-t5-base-msmarco
  - doc2query
tags: []
---

## Description

docT5query is the document-to-query generation model used in [[du-2024-bottleneckminimal-2405-10974]] to synthesize `GenQ` queries for query-aware indexing. The authors also fine-tune it on NQ320K to improve the estimate of `\mu_{Q|x}` and raise final retrieval accuracy.

## Key Contributions

- Produces synthetic queries that help approximate each document's query distribution for BMI.
- After fine-tuning on NQ320K, enables the improved BMI configuration that reaches `67.8` Rec@1 on that benchmark.

## Related Concepts

- [[generative-retrieval]]
- [[data-augmentation]]
- [[k-means-clustering]]

## Sources

- [[du-2024-bottleneckminimal-2405-10974]]
