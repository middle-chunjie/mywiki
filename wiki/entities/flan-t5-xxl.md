---
type: entity
title: FlanT5-XXL
slug: flan-t5-xxl
date: 2026-04-20
entity_type: tool
aliases: [Flan-T5 XXL, Flan T5 XXL]
tags: []
---

## Description

FlanT5-XXL is the frozen `11B` instruction-tuned language model used for generation in [[salemi-2024-optimization]]. The paper keeps this model fixed and optimizes only retrieval and retriever selection components around it.

## Key Contributions

- Serves as the downstream generator for all LaMP experiments.
- Makes it possible to isolate the effect of retrieval optimization from generator fine-tuning.
- Is decoded with beam search using beam size `4` in the reported experiments.

## Related Concepts

- [[large-language-model]]
- [[personalization]]
- [[retrieval-augmented-generation]]

## Sources

- [[salemi-2024-optimization]]
