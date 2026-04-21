---
type: entity
title: DeepSeek 67B
slug: deepseek-67b
date: 2026-04-20
entity_type: tool
aliases: [DeepSeek-67B, DeepSeek LLM 67B]
tags: []
---

## Description

DeepSeek 67B is the dense predecessor used as the main internal baseline throughout the paper. It anchors comparisons for training cost, KV-cache size, and multilingual benchmark quality against DeepSeek-V2.

## Key Contributions

- Serves as the direct baseline showing the efficiency gains of moving from a dense model to the DeepSeek-V2 MoE design.
- Provides the earlier tokenizer and data-processing pipeline reused by DeepSeek-V2.
- Establishes the cost and throughput reference points that make the reported `42.5%` and `5.76x` improvements meaningful.

## Related Concepts

- [[large-language-model]]
- [[transformer]]
- [[byte-pair-encoding]]

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]
