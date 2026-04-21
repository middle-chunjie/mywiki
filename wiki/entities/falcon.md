---
type: entity
title: Falcon
slug: falcon
date: 2026-04-20
entity_type: model
aliases: [Falcon-7B, Falcon-40B]
tags: []
---

## Description

Falcon is one of the evaluated autoregressive model families in the paper. Its results help show that StreamingLLM generalizes across model scales and architectures beyond Llama-2.

## Key Contributions

- Serves as one of the cross-family validation targets in the long-stream PG19 experiments.
- Shows perplexity improvement from `17.90` to `12.12` when four sink tokens are preserved.

## Related Concepts

- [[attention-sink]]
- [[kv-cache]]
- [[window-attention]]

## Sources

- [[xiao-2024-efficient-2309-17453]]
