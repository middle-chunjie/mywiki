---
type: entity
title: DeepSeek-V3
slug: deepseek-v3
date: 2026-04-20
entity_type: tool
aliases: [DeepSeek V3]
tags: []
---

## Description

DeepSeek-V3 is one of the backbone language models used in [[wang-2026-ragrouterbench-2602-00296]] for generation and for knowledge-triplet extraction during corpus preprocessing.

## Key Contributions

- Serves as the primary generator in the benchmark's shared evaluation setup.
- Produces knowledge triplets at temperature `0.0` for graph construction.
- Delivers higher benchmark scores than [[llama-3-1-8b]] across the reported dataset averages.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[adaptive-rag-routing]]
- [[graph-rag]]

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]
