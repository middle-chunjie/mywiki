---
type: entity
title: Attention Sort
slug: attention-sort
date: 2026-04-20
entity_type: tool
aliases: [attention sorting]
tags: []
---

## Description

Attention Sort is the long-context baseline compared against in [[merth-2024-superposition-2404-06910]], where retrieved documents are reordered but still jointly attended during generation.

## Key Contributions

- Provides a recency-bias mitigation baseline for long-context RAG without path pruning or path caching.
- On NaturalQuestions-Open with `mpt-7b-instruct`, it reaches `0.028` accuracy and only `0.3x` theoretical speedup, far below superposition prompting.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[long-context-inference]]
- [[prompt-engineering]]

## Sources

- [[merth-2024-superposition-2404-06910]]
