---
type: entity
title: Llama-3.1-8B-Instruct
slug: llama-3.1-8b-instruct
date: 2026-04-20
entity_type: tool
aliases: [Llama 3.1 8B Instruct]
tags: []
---

## Description

Llama-3.1-8B-Instruct is the base model that the paper fine-tunes to obtain CollabLLM variants. It also serves as the main baseline and the backbone for the proactive-prompting baseline.

## Key Contributions

- Provides the pretrained instruction-following policy from which SFT, DPO, and PPO variants are derived.
- Supplies the reference baseline used to quantify gains in task performance, efficiency, and interactivity.

## Related Concepts

- [[lora]]
- [[direct-preference-optimization]]
- [[proximal-policy-optimization]]

## Sources

- [[wu-2025-collabllm-2502-00640]]
