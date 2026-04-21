---
type: entity
title: YaRN
slug: yarn
date: 2026-04-20
entity_type: tool
aliases: [Yarn]
tags: []
---

## Description

YaRN is the long-context extension method applied after DeepSeek-V2 pretraining to expand the model from `4K` to `128K` context. In this paper it is attached specifically to the RoPE-carrying shared key inside MLA.

## Key Contributions

- Extends DeepSeek-V2's usable context window without retraining the full model from scratch.
- Is configured with `s = 40`, `alpha = 1`, `beta = 32`, and target length `160K` in the reported setup.
- Enables the long-context Needle-In-A-Haystack evaluation reported up to `128K`.

## Related Concepts

- [[long-context-training]]
- [[rotary-positional-embedding]]
- [[key-value-cache]]

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]
