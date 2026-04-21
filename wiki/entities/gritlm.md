---
type: entity
title: GritLM
slug: gritlm
date: 2026-04-20
entity_type: tool
aliases: [GritLM, GritLM 7B, GritLM 8x7B]
tags: []
---

## Description

GritLM is the unified model family introduced in [[muennighoff-2024-generative-2402-09906]] to perform both text embedding and generation with a single instruction-tuned LLM.

## Key Contributions

- Achieves `66.8` MTEB average at 7B while preserving strong generative performance.
- Supports both retrieval-stage encoding and generation-stage reranking within one model family.
- Enables caching-based RAG variants that reuse representations to reduce long-context latency.

## Related Concepts

- [[generative-representational-instruction-tuning]]
- [[representational-instruction-tuning]]
- [[retrieval-augmented-generation]]

## Sources

- [[muennighoff-2024-generative-2402-09906]]
