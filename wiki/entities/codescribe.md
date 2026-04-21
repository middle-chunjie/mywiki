---
type: entity
title: CODESCRIBE
slug: codescribe
date: 2026-04-20
entity_type: tool
aliases: [CODE-SCRIBE]
tags: []
---

## Description

CODESCRIBE is the model proposed in [[guo-2022-modeling]] for source code summarization. It combines a Transformer code encoder, a residual GraphSAGE AST encoder with triplet positions, and a multi-source pointer-generator decoder.

## Key Contributions

- Introduces triplet-position-aware AST encoding for preserving ordered hierarchical syntax.
- Fuses AST and token-sequence information in the decoder instead of relying on only one source.
- Achieves state-of-the-art results on the Java and Python benchmarks reported in the paper.

## Related Concepts

- [[triplet-position]]
- [[graphsage]]
- [[pointer-generator-network]]

## Sources

- [[guo-2022-modeling]]
