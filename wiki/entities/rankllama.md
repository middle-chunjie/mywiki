---
type: entity
title: RankLLaMA
slug: rankllama
date: 2026-04-20
entity_type: tool
aliases: [RankLLaMA reranker]
tags: []
---

## Description

RankLLaMA is the pointwise LLaMA-based reranker introduced in [[ma-2023-finetuning-2310-08319]]. It scores each query-document pair with a scalar head over the final `</s>` representation.

## Key Contributions

- Reorders candidates from RepLLaMA with parallel pointwise scoring instead of generative decoding.
- Achieves the best passage and document reranking results among the baselines reported in the paper.
- Shows that `13B` scaling offers a small gain on passage reranking but not on zero-shot BEIR.

## Related Concepts

- [[reranking]]
- [[pointwise-ranking]]
- [[large-language-model]]

## Sources

- [[ma-2023-finetuning-2310-08319]]
