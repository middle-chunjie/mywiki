---
type: entity
title: GPT2-XL
slug: gpt2-xl
date: 2026-04-20
entity_type: model
aliases: [GPT-2 XL, GPT2 XL]
tags: []
---

## Description

GPT2-XL is the primary `1.5B`-parameter decoder-only model analyzed in [[wang-2023-label]]. The paper uses it for saliency analysis, causal isolation, anchor re-weighting, compression, and confusion diagnosis.

## Key Contributions

- Serves as the main model for validating shallow-layer information aggregation and deep-layer anchor extraction.
- Provides the only model used in the anchor re-weighting experiments due to compute limits.
- Supports the TREC confusion analysis based on anchor key-vector distances.

## Related Concepts

- [[large-language-model]]
- [[in-context-learning]]
- [[label-anchor]]

## Sources

- [[wang-2023-label]]
