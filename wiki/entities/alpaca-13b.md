---
type: entity
title: Alpaca 13B
slug: alpaca-13b
date: 2026-04-20
entity_type: tool
aliases: [Alpaca-13B, Alpaca 13B, Alpaca]
tags: []
---

## Description

Alpaca 13B is one of the smaller instruction-following language models evaluated in [[wang-2023-selfprompted-2310-13552]]. The paper uses it to test whether SP-CoT transfers beyond proprietary frontier APIs.

## Key Contributions

- Serves as a small-model stress test for self-prompted reasoning in ODMR.
- Improves from `8.9` to `17.4` mean EM under SP-CoT, an `+8.5` gain over zero-shot prompting.
- Helps show that generated CoT demonstrations can substantially narrow the gap between `13B` and much larger instruction-following models.

## Related Concepts

- [[large-language-model]]
- [[chain-of-thought]]
- [[self-prompting]]

## Sources

- [[wang-2023-selfprompted-2310-13552]]
