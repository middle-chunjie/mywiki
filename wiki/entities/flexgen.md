---
type: entity
title: FlexGen
slug: flexgen
date: 2026-04-20
entity_type: tool
aliases: [FlexGen inference system]
tags: []
---

## Description

FlexGen is a high-throughput LLM inference system that supports CPU/GPU offloading to enable large-batch generation on commodity hardware, developed by Ying Sheng, Lianmin Zheng, and collaborators.

## Key Contributions

- Baseline inference system on which H2O is implemented, providing the offloading and batching infrastructure.
- Used as a primary throughput comparison baseline (H2O achieves up to 3× higher throughput vs FlexGen on OPT-6.7B/30B).

## Related Concepts

- [[kv-cache]]
- [[large-language-model]]
- [[autoregressive-decoding]]

## Sources

- [[zhang-2023-ho-2306-14048]]
