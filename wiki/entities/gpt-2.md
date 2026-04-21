---
type: entity
title: GPT-2
slug: gpt-2
date: 2026-04-20
entity_type: tool
aliases: [GPT-2, GPT2]
tags: []
---

## Description

GPT-2 is the decoder-only Transformer architecture used as the backbone for the three task-specific models in [[izadi-2022-codefill-2202-06689]]. The paper adopts GPT-2-style blocks for autoregressive code modeling rather than treating code completion as a bidirectional masked-language task.

## Key Contributions

- Provides the causal self-attention architecture behind token-value, token-type, and statement-completion decoding.
- Supports long-context autoregressive prediction with the paper's maximum sequence length of `2048`.

## Related Concepts

- [[transformer]]
- [[decoder-only-language-model]]
- [[language-modeling]]
- [[code-completion]]

## Sources

- [[izadi-2022-codefill-2202-06689]]
