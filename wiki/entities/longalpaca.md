---
type: entity
title: LongAlpaca
slug: longalpaca
date: 2026-04-20
entity_type: dataset
aliases: [Long Alpaca, LongAlpaca-16K, LongAlpaca-4K]
tags: []
---

## Description

LongAlpaca is a training dataset and model series for long-context instruction-following, produced by the LongLoRA paper (Chen et al., 2023). It extends the Alpaca instruction-tuning format to longer sequences (up to 16K tokens) and is used both as training data and as a fine-tuning baseline in long-context LLM research.

## Key Contributions

- Provides 10K long-instruction samples (up to 16K tokens) used as supplementary training data for Activation Beacon alongside RedPajama.
- LongAlpaca-16K serves as a fine-tuned full-attention baseline in the Activation Beacon paper, achieving competitive perplexity on PG19/Proof-Pile/CodeParrot at 16K but becoming OOM beyond 32K.
- Demonstrates that LongLoRA-style fine-tuning with shifted sparse attention can be comparable to full attention on long-context tasks.

## Related Concepts

- [[context-length-extension]]
- [[long-context-modeling]]
- [[long-context-training]]

## Sources

- [[zhang-2024-long-2401-03462]]
