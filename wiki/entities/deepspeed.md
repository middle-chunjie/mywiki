---
type: entity
title: DeepSpeed
slug: deepspeed
date: 2026-04-20
entity_type: tool
aliases: [DeepSpeed ZeRO]
tags: []
---

## Description

DeepSpeed is the distributed training stack used in [[breton-2025-neobert-2502-19587]] to scale NeoBERT pretraining across devices. The paper specifically highlights ZeRO-based memory savings and larger feasible batch sizes.

## Key Contributions

- Enables the `2M`-token batch size used during NeoBERT pretraining.
- Reduces duplicated optimizer and activation memory through ZeRO-style partitioning.

## Related Concepts

- [[long-context-training]]
- [[model-scaling]]
- [[masked-language-modeling]]

## Sources

- [[breton-2025-neobert-2502-19587]]
