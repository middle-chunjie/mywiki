---
type: entity
title: GPT-J
slug: gpt-j
date: 2026-04-20
entity_type: model
aliases: [GPT-J-6B]
tags: []
---

## Description

GPT-J is a public `6B` autoregressive language model used in this paper as a non-meta-trained baseline for in-context learning analysis. It is one of the strongest sub-`13B` baselines in the study's ablations.

## Key Contributions

- Provides direct and channel prompting baselines for measuring the effect of random labels, OOD inputs, and format changes.
- Shows that some settings remain robust to incorrect labels, while GPT-J classification can still incur nearly `10%` loss when all labels are wrong.

## Related Concepts

- [[large-language-model]]
- [[few-shot-prompting]]
- [[input-label-mapping]]

## Sources

- [[min-2022-rethinking-2202-12837]]
