---
type: entity
title: WizardLM 13B
slug: wizardlm-13b
date: 2026-04-20
entity_type: tool
aliases: [WizardLM-13B, WizardLM 13B, WizardLM]
tags: []
---

## Description

WizardLM 13B is one of the instruction-tuned `13B` models evaluated in [[wang-2023-selfprompted-2310-13552]] to test the generality of SP-CoT on smaller open models.

## Key Contributions

- Provides one of the strongest small-model test beds in the paper's transfer study.
- Improves from `10.9` to `22.5` mean EM with SP-CoT, the largest reported gain (`+11.6`) among the three `13B` models.
- Supports the claim that structured self-generated demonstrations can reliably improve instruction-tuned open models on ODMR.

## Related Concepts

- [[large-language-model]]
- [[chain-of-thought]]
- [[self-prompting]]

## Sources

- [[wang-2023-selfprompted-2310-13552]]
