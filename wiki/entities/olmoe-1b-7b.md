---
type: entity
title: OLMoE-1B-7B
slug: olmoe-1b-7b
date: 2026-04-20
entity_type: tool
aliases: [OLMoE 1B-7B]
tags: []
---

## Description

OLMoE-1B-7B is the third MoE language model evaluated in [[li-2024-your-2410-10814]]. The paper reports `16` layers with `64` experts per layer and uses it to show that MoEE transfers across different MoE model scales and training lineages.

## Key Contributions

- Provides the highest PromptEOL MTEB average in the paper at `55.16` with MoEE (sum).
- Supports the claim that combining RW and HS is not specific to a single MoE implementation.

## Related Concepts

- [[mixture-of-experts]]
- [[routing-weight-embedding]]
- [[training-free-embedding]]

## Sources

- [[li-2024-your-2410-10814]]
