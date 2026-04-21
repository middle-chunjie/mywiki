---
type: entity
title: Atom-7B
slug: atom-7b
date: 2026-04-20
entity_type: tool
aliases: [Atom 7B, Llama2-Chinese Atom, Atom-7B-chat, Atom-7B-CP]
tags: []
---

## Description

Atom-7B is an open-source Chinese-extended Llama-2 model developed by FlagAlpha (Llama2-Chinese project). Atom-7B-chat is the instruction-following variant; Atom-7B-CP is a version further fine-tuned on domain-specific (cloud product) documents for use in KnowPAT.

## Key Contributions

- Serves as the backbone LLM in KnowPAT for cloud-product domain QA experiments.
- Extended Llama-2 with Chinese vocabulary, enabling effective Chinese-language QA fine-tuning.
- Atom-7B-CP (domain-continued pretraining) outperforms vanilla Atom-7B zero-shot on all metrics and serves as the backbone for all fine-tuning and alignment baselines.

## Related Concepts

- [[large-language-model]]
- [[supervised-fine-tuning]]
- [[preference-alignment]]
- [[domain-specific-question-answering]]

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]
