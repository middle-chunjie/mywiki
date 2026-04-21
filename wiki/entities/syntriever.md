---
type: entity
title: Syntriever
slug: syntriever
date: 2026-04-20
entity_type: tool
aliases: [Syntriever framework]
tags: []
---

## Description

Syntriever is the two-stage retriever training framework introduced in [[kim-2025-syntriever-2502-03824]] for distilling and aligning dense retrievers with black-box LLMs.

## Key Contributions

- Combines synthetic query and passage generation with hallucination-aware relabeling in a distillation stage.
- Uses partial Plackett-Luce preference modeling to align retrievers with LLM ranking judgments.
- Delivers state-of-the-art retrieval results across multiple BeIR benchmark settings.

## Related Concepts

- [[dense-retrieval]]
- [[knowledge-distillation]]
- [[plackett-luce-ranking]]

## Sources

- [[kim-2025-syntriever-2502-03824]]
