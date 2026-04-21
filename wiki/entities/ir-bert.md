---
type: entity
title: IR-BERT
slug: ir-bert
date: 2026-04-20
entity_type: tool
aliases: [IR BERT, IRBERT]
tags: []
---

## Description

IR-BERT is the pretrained LLVM-IR encoder used in [[gui-2022-crosslanguage-2201-07420]] to initialize XLIR. It is trained as a masked language model over a large external LLVM-IR corpus before fine-tuning on code matching tasks.

## Key Contributions

- Supplies the pretrained encoder parameters used by XLIR.
- Is trained on `48,023,781` LLVM-IR instructions from `855,792` functions collected from `11` real-world software systems.
- Uses masked language modeling with BPE tokenization over LLVM-IR sequences.

## Related Concepts

- [[masked-language-modeling]]
- [[llvm-ir]]
- [[byte-pair-encoding]]

## Sources

- [[gui-2022-crosslanguage-2201-07420]]
