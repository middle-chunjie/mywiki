---
type: entity
title: RETRO
slug: retro
date: 2026-04-20
entity_type: model
aliases: [Retrieval-Enhanced Transformer]
tags: []
---

## Description

RETRO is the retrieval-enhanced Transformer introduced in [[borgeaud-2022-improving-2112-04426]]. It augments autoregressive language modeling with chunk-level retrieval from an external datastore.

## Key Contributions

- Shows that trillion-token external memory can rival much larger purely parametric language models.
- Introduces chunked cross-attention as a scalable way to integrate retrieved neighbors into left-to-right decoding.

## Related Concepts

- [[retrieval-based-language-model]]
- [[semi-parametric-language-model]]
- [[chunked-cross-attention]]

## Sources

- [[borgeaud-2022-improving-2112-04426]]
