---
type: entity
title: FlashAttention
slug: flashattention
date: 2026-04-20
entity_type: tool
aliases: [FlashAttention-2]
tags: []
---

## Description

FlashAttention is the exact attention kernel used in [[breton-2025-neobert-2502-19587]] to reduce memory traffic and improve long-sequence efficiency. It is part of the systems stack that lets NeoBERT remain fast at `4,096`-token context lengths.

## Key Contributions

- Avoids materializing full attention matrices, lowering the memory cost of long-context encoder inference and training.
- Supports NeoBERT's throughput advantage over ModernBERT on long inputs.

## Related Concepts

- [[flash-attention]]
- [[long-context-training]]
- [[length-extrapolation]]

## Sources

- [[breton-2025-neobert-2502-19587]]
