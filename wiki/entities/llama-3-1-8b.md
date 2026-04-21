---
type: entity
title: LLaMA-3.1-8B
slug: llama-3-1-8b
date: 2026-04-20
entity_type: tool
aliases: [LLaMA 3.1 8B, Llama 3.1 8B]
tags: []
---

## Description

LLaMA-3.1-8B is the smaller backbone language model used in [[wang-2026-ragrouterbench-2602-00296]] to measure how RAG routing behavior changes under a weaker generator.

## Key Contributions

- Provides a second backbone for the benchmark's controlled model comparison.
- Shows substantially lower LLM-as-a-Judge accuracy than [[deepseek-v3]] on most settings.
- Helps demonstrate that routing conclusions depend on both retrieval paradigm and generator capability.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[adaptive-rag-routing]]
- [[llm-as-a-judge]]

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]
