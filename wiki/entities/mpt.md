---
type: entity
title: MPT
slug: mpt
date: 2026-04-20
entity_type: model
aliases: [MPT-7B, MPT-30B]
tags: []
---

## Description

MPT is one of the evaluated open LLM families in the paper and serves as the main ALiBi-based testbed for StreamingLLM. It is used to show that the method is not specific to RoPE-based models.

## Key Contributions

- Demonstrates StreamingLLM compatibility with [[alibi]] positional biasing.
- Shows perplexity recovery from `460.29` under window attention to about `14.99` with four sink tokens on MPT-7B.

## Related Concepts

- [[alibi]]
- [[length-extrapolation]]
- [[window-attention]]

## Sources

- [[xiao-2024-efficient-2309-17453]]
