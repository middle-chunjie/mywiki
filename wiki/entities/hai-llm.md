---
type: entity
title: HAI-LLM
slug: hai-llm
date: 2026-04-20
entity_type: tool
aliases: [HAI LLM]
tags: []
---

## Description

HAI-LLM is the internal DeepSeek training framework used for all experiments in the paper. It provides the distributed systems substrate for pipeline parallelism, expert parallelism, evaluation, and long-context adaptation.

## Key Contributions

- Hosts the `16`-way zero-bubble pipeline and `8`-way expert-parallel training recipe for DeepSeek-V2.
- Integrates the internal evaluation framework used for bilingual, code, math, and chat benchmarks.
- Supports communication overlap and custom kernels needed to keep the sparse model efficient in practice.

## Related Concepts

- [[expert-parallelism]]
- [[long-context-training]]
- [[large-language-model]]

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]
