---
type: entity
title: Contriever
slug: contriever
date: 2026-04-20
entity_type: tool
aliases: [Contriever retriever]
tags: []
---

## Description

Contriever is the unsupervised dense retriever used in the paper to test how retrieval granularity affects passage recall and downstream QA.

## Key Contributions

- Shows strong gains from proposition indexing, with average Recall@20 rising from `62.8` to `70.5`.
- Improves FiD top-20 EM from `33.7` to `38.7` and LLaMA-2-7B average `EM@500` from `34.1` to `37.3`.

## Related Concepts

- [[dense-retrieval]]
- [[proposition-retrieval]]
- [[cross-task-generalization]]

## Sources

- [[chen-2024-dense-2312-06648]]
