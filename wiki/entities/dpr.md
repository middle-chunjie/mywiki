---
type: entity
title: DPR
slug: dpr
date: 2026-04-20
entity_type: tool
aliases: [Dense Passage Retriever]
tags: []
---

## Description

DPR is the supervised dense retriever evaluated in the paper as a passage-trained baseline for comparing passage, sentence, and proposition indexing.

## Key Contributions

- Shows smaller average gains than unsupervised retrievers but still improves average Recall@20 from `69.5` to `72.2` with proposition indexing.
- Demonstrates especially visible benefits on out-of-training-distribution datasets such as EntityQuestions.

## Related Concepts

- [[dense-retrieval]]
- [[passage-retrieval]]
- [[cross-task-generalization]]

## Sources

- [[chen-2024-dense-2312-06648]]
