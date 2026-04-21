---
type: entity
title: InfiniteBench
slug: infinitebench
date: 2026-04-20
entity_type: dataset
aliases: [InfiniteBench, InftyBench]
tags: []
---

## Description

InfiniteBench is a long-context benchmark used in [[qian-2024-memorag-2409-05591]] to evaluate QA, summarization, and long-book understanding tasks whose contexts can exceed typical model windows by a large margin.

## Key Contributions

- Supplies evaluation tasks such as `En.SUM` and `En.QA` for testing MemoRAG beyond standard passage-level QA.
- Helps measure whether memory-guided retrieval remains effective when contexts grow to hundreds of thousands of tokens.

## Related Concepts

- [[context-window]]
- [[query-focused-summarization]]
- [[multihop-question-answering]]

## Sources

- [[qian-2024-memorag-2409-05591]]
