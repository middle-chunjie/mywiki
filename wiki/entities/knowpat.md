---
type: entity
title: KnowPAT
slug: knowpat
date: 2026-04-20
entity_type: tool
aliases: [Knowledgeable Preference AlignmenT, KnowPAT framework]
tags: []
---

## Description

KnowPAT (Knowledgeable Preference AlignmenT) is an open-source LLM fine-tuning framework for domain-specific QA that integrates knowledge graph retrieval with preference alignment. The code is released at https://github.com/zjukg/KnowPAT; the proprietary cloud-product dataset cannot be released.

## Key Contributions

- Introduces the first preference alignment framework specifically designed for domain-specific QA with external KGs.
- Achieves BLEU-4 of `12.11` (+43.99% over best baseline) and CIDEr of `54.86` (+66.04%) on Huawei cloud-product QA test set.
- Combines style and knowledge preference sets in a unified multi-task training objective with adaptive weighting.

## Related Concepts

- [[preference-alignment]]
- [[knowledge-preference-alignment]]
- [[style-preference-alignment]]
- [[domain-specific-question-answering]]
- [[knowledge-graph]]

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]
