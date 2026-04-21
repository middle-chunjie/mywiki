---
type: entity
title: Alpaca
slug: alpaca
date: 2026-04-20
entity_type: model
aliases: [Stanford Alpaca]
tags: []
---

## Description

Alpaca is the instruction-tuned LLaMA-based model used in [[yuan-2023-rrhf-2304-05302]] as the main initialization for RRHF, PPO, and SFT comparisons.

## Key Contributions

- Serves as the strongest base checkpoint in the paper's main HH experiments, where RRHF reaches reward `-0.96` from Alpaca initialization.
- Provides the prompt set for the Wombat experiment when RRHF is trained with ChatGPT-scored responses.

## Related Concepts

- [[large-language-model]]
- [[instruction-tuning]]
- [[supervised-fine-tuning]]

## Sources

- [[yuan-2023-rrhf-2304-05302]]
