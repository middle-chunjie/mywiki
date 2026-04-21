---
type: entity
title: NVIDIA H800
slug: nvidia-h800
date: 2026-04-20
entity_type: tool
aliases: [H800, NVIDIA H800 GPU]
tags: []
---

## Description

NVIDIA H800 is the accelerator used for both training and deployment measurements in the paper. DeepSeek reports DeepSeek-V2 results on H800 clusters with `8` GPUs per node and uses the same hardware family for the single-node throughput numbers.

## Key Contributions

- Provides the hardware platform for the reported `172.8K` GPU-hours-per-trillion-token training cost.
- Underlies the single-node serving result of more than `50K` generated tokens/s and `100K` prompt tokens/s.
- Makes the paper's systems claims concrete by tying MLA and sparse routing gains to a specific deployment target.

## Related Concepts

- [[expert-parallelism]]
- [[key-value-cache]]
- [[large-language-model]]

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]
