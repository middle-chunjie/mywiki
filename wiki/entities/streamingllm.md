---
type: entity
title: StreamingLLM
slug: streamingllm
date: 2026-04-20
entity_type: tool
aliases: [Streaming LLM]
tags: []
---

## Description

StreamingLLM is the inference framework introduced by the paper for stable long-stream decoding in pretrained autoregressive language models. It keeps a few initial attention-sink tokens together with a rolling recent-token KV cache.

## Key Contributions

- Restores stable perplexity in long-stream decoding without model fine-tuning.
- Achieves up to `22.2x` per-token speedup over sliding-window recomputation.
- Supports streaming evaluation beyond `4 million` tokens across several LLM families.

## Related Concepts

- [[attention-sink]]
- [[kv-cache]]
- [[window-attention]]

## Sources

- [[xiao-2024-efficient-2309-17453]]
