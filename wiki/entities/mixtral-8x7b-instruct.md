---
type: entity
title: Mixtral 8x7B Instruct
slug: mixtral-8x7b-instruct
date: 2026-04-20
entity_type: tool
aliases: [Mixtral 8X7B Instruct]
tags: []
---

## Description

Mixtral 8x7B Instruct is the instruction-tuned generator used inside the mtRAG annotation interface and for the paper's contextual [[query-rewriting]] reference implementation.

## Key Contributions

- Produces draft answers that human annotators repair while constructing benchmark conversations.
- Rewrites non-standalone user turns into standalone queries for retrieval experiments.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[query-rewriting]]
- [[multi-turn-conversation]]

## Sources

- [[katsis-2025-mtrag-2501-03468]]
