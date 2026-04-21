---
type: entity
title: PIC-LM
slug: pic-lm
date: 2026-04-20
entity_type: model
aliases: [PIC LM, Llama-3.1-PIC-LM-8B]
tags: []
---

## Description

PIC-LM is the `8B` post-trained language model introduced in [[he-2025-precise-2506-06589]], built from Llama 3.1 8B Instruct to improve claim-faithful long-form generation.

## Key Contributions

- Combines PIC-formatted supervised fine-tuning with length-normalized DPO on weakly supervised preference pairs.
- Reaches `91.0%` average `F_1` on full PIC and `93.3%` average precision on partial PIC.
- Improves downstream factual generation on ASQA, birthplace fact checking, and QAMParI.

## Related Concepts

- [[precise-information-control]]
- [[supervised-fine-tuning]]
- [[direct-preference-optimization]]

## Sources

- [[he-2025-precise-2506-06589]]
