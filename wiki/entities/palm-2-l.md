---
type: entity
title: PaLM 2-L
slug: palm-2-l
date: 2026-04-20
entity_type: tool
aliases: [PaLM 2 L, PaLM2-L]
tags: []
---

## Description

PaLM 2-L is the primary scorer model in [[yang-2024-large-2309-03409]] and is also evaluated as both a pretrained and instruction-tuned optimizer within OPRO.

## Key Contributions

- Serves as the scorer for the main GSM8K prompt-optimization results, where the best OPRO-discovered instruction reaches `80.2%` test accuracy.
- Is itself used as an optimizer, reaching `79.9%` test accuracy on GSM8K when optimizing prompts for PaLM 2-L scoring.

## Related Concepts

- [[large-language-model]]
- [[prompt-optimization]]
- [[zero-shot-prompting]]

## Sources

- [[yang-2024-large-2309-03409]]
