---
type: entity
title: IIRC
slug: iirc
date: 2026-04-20
entity_type: dataset
aliases: [Incomplete Information Reading Comprehension]
tags: []
---

## Description

IIRC is the incomplete-information reading comprehension benchmark used in [[su-2024-dragin-2403-10081]] to test dynamic retrieval on question answering that requires consulting evidence beyond the initial passage. In the paper's setup, questions without answers are removed, leaving `954` evaluation examples.

## Key Contributions

- Serves as the paper's reading-comprehension benchmark alongside the multihop and commonsense QA datasets.
- Provides the testbed for the retrieval-timing ablation, where DRAGIN outperforms FLARE, FL-RAG, and FS-RAG for both evaluated model families.

## Related Concepts

- [[knowledge-intensive-generation]]
- [[retrieval-augmented-generation]]
- [[large-language-model]]

## Sources

- [[su-2024-dragin-2403-10081]]
