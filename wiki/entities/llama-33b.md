---
type: entity
title: LLaMA 33B
slug: llama-33b
date: 2026-04-20
entity_type: tool
aliases: [Llama 33B, LLaMA-33B]
tags: []
---

## Description

LLaMA 33B is the backend large language model used in [[feng-2023-retrievalgeneration-2310-05149]] for both intermediate document generation and final answer prediction. The paper selects it as a balance between performance and computational cost.

## Key Contributions

- Provides the generative backbone for both refine and refresh variants of ITRG.
- Is evaluated in `0`-shot, `1`-shot, and `5`-shot QA settings against GPT-3.5 baselines.

## Related Concepts

- [[large-language-model]]
- [[retrieval-augmented-generation]]
- [[in-context-learning]]

## Sources

- [[feng-2023-retrievalgeneration-2310-05149]]
