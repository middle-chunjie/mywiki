---
type: entity
title: ProLong
slug: prolong
date: 2026-04-20
entity_type: tool
aliases: [ProLong-8B]
tags: []
---

## Description

ProLong is the long-context `8B` language model introduced in [[gao-2024-how-2410-02660]]. It is initialized from Llama-3-8B-Instruct, continually trained for `40B` tokens, and supports contexts up to `512K`.

## Key Contributions

- Demonstrates the paper's final long-context training recipe combining balanced long/short data, RoPE scaling, document masking, and short-context SFT.
- Achieves `49.4` average on HELMET at `128K`, outperforming other open models of similar scale.
- Shows effective scaling beyond `128K`, with QA improving from `31.7` at `32K` to `49.7` at `512K`.

## Related Concepts

- [[long-context-training]]
- [[supervised-fine-tuning]]
- [[sequence-parallelism]]

## Sources

- [[gao-2024-how-2410-02660]]
