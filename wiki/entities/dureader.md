---
type: entity
title: DuReader
slug: dureader
date: 2026-04-20
entity_type: dataset
aliases: [Du Reader]
tags: []
---

## Description

DuReader is the Chinese multi-document question answering dataset used in [[ye-2024-rag-2406-13249]] to test R2AG under long-context conditions. In this paper, each query is paired with `20` candidate documents and about `16k` tokens of context.

## Key Contributions

- Serves as the paper's Chinese evaluation benchmark for long-context retrieval-augmented generation.
- Shows modest frozen-model gains for R2AG (`0.1510` F1 vs `0.1395` for the Qwen1.5-0.5B baseline) and stronger gains when combined with RAFT.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[multihop-question-answering]]
- [[large-language-model]]

## Sources

- [[ye-2024-rag-2406-13249]]
