---
type: entity
title: Qwen2.5-Coder
slug: qwen2-5-coder
date: 2026-04-20
entity_type: tool
aliases: [Qwen2.5 Coder, Qwen2.5-Coder-0.5B, Qwen2.5-Coder-1.5B]
tags: []
---

## Description

Qwen2.5-Coder is the pretrained code-generation model family used as the backbone for the `0.5B` and `1.5B` `jina-code-embeddings` variants in [[kryvosheieva-2025-efficient-2508-21290]]. The paper adapts these decoder-only code LLMs into retrieval-oriented embedding models.

## Key Contributions

- Supplies the pretrained text-and-code backbone from which the paper derives compact code embedders.
- Enables the paper's last-token-pooling embedding strategy on decoder-only hidden states.
- Demonstrates that code-generation pretraining can transfer effectively to embedding-based retrieval.

## Related Concepts

- [[decoder-only-language-model]]
- [[last-token-pooling]]
- [[code-embedding]]

## Sources

- [[kryvosheieva-2025-efficient-2508-21290]]
