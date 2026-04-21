---
type: entity
title: GSM8K
slug: gsm8k
date: 2026-04-20
entity_type: benchmark
aliases: [Grade School Math 8K]
tags: [benchmark, math, reasoning]
---

## Description

GSM8K is the natural-language mathematics benchmark used in [[gloeckle-2024-better-2404-19737]] to test how multi-token-pretrained language models behave on few-shot reasoning.

## Key Contributions

- Supplies the `8`-shot evaluation setting used to compare next-token and multi-token pretraining after `200B` and `500B` language tokens.
- Shows that `n = 2` helps at `200B` tokens, while the advantage reverses after `500B` and `n = 4` remains worse throughout.

## Related Concepts

- [[algorithmic-reasoning]]
- [[multi-token-prediction]]
- [[large-language-model]]

## Sources

- [[gloeckle-2024-better-2404-19737]]
