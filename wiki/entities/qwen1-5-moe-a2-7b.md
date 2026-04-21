---
type: entity
title: Qwen1.5-MoE-A2.7B
slug: qwen1-5-moe-a2-7b
date: 2026-04-20
entity_type: tool
aliases: [Qwen1.5-MoE A2.7B]
tags: []
---

## Description

Qwen1.5-MoE-A2.7B is one of the MoE backbones evaluated in [[li-2024-your-2410-10814]]. The paper reports `24` layers and `60` experts per layer for this model and uses it to test whether MoEE generalizes beyond DeepSeekMoE.

## Key Contributions

- Confirms that MoEE improves over standalone HS and RW on a second MoE architecture family.
- Reaches an MTEB average of `42.25` without prompts and `55.04` with PromptEOL under MoEE (sum).

## Related Concepts

- [[mixture-of-experts]]
- [[training-free-embedding]]
- [[semantic-textual-similarity]]

## Sources

- [[li-2024-your-2410-10814]]
