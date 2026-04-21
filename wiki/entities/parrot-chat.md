---
type: entity
title: Parrot-Chat
slug: parrot-chat
date: 2026-04-20
entity_type: tool
aliases: [PARROT Chat]
tags: []
---

## Description

Parrot-Chat is the final chat model trained in [[sun-2024-parrot-2310-07301]] using the Parrot-40K corpus. It is designed to improve long-turn instruction following among open-source `13B` chat models.

## Key Contributions

- Achieves `83.42` Alpaca-Eval, `6.81` MT-Bench, and `6.56` MT-Bench++ in the paper.
- Outperforms Vicuna v1.5 on all three reported open-source comparisons while using only `40K` supervised dialogues.
- Shows more stable performance on long conversations than baselines trained on shorter sessions.

## Related Concepts

- [[instruction-following]]
- [[supervised-fine-tuning]]
- [[large-language-model]]

## Sources

- [[sun-2024-parrot-2310-07301]]
