---
type: entity
title: OPT
slug: opt-model
date: 2026-04-20
entity_type: tool
aliases: [OPT, Open Pre-trained Transformer, OPT-6.7B, OPT-13B, OPT-30B, OPT-66B, OPT-175B]
tags: []
---

## Description

OPT (Open Pre-trained Transformer) is a suite of open-source autoregressive language models from Meta AI ranging from 125M to 175B parameters, trained on a publicly released corpus and evaluated on standard NLP benchmarks.

## Key Contributions

- Primary evaluation model family for H2O, used across sizes 6.7B to 175B to validate the KV-cache eviction approach.
- The 6.7B and 30B variants are used for throughput benchmarking on T4 and A100 GPUs.

## Related Concepts

- [[large-language-model]]
- [[autoregressive-decoding]]
- [[kv-cache]]

## Sources

- [[zhang-2023-ho-2306-14048]]
