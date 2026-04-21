---
type: entity
title: ColBERT-X
slug: colbert-x
date: 2026-04-20
entity_type: tool
aliases: [ColBERT X, multilingual ColBERT]
tags: []
---

## Description

ColBERT-X is the multilingual late-interaction dual-encoder retrieval model used in [[yang-2024-distillation-2405-00977]] as the student architecture. The paper fine-tunes it from `XLM-RoBERTa large` and trains it with multilingual translation plus distillation.

## Key Contributions

- Serves as the student retriever that receives teacher supervision from English `ColBERTv2` and `MonoT5`.
- Outperforms prior `ColBERT-X MTT` training when optimized with [[multilingual-translate-distill]].
- Supports multilingual passage ranking while preserving late-interaction scoring behavior.

## Related Concepts

- [[multilingual-dense-retrieval]]
- [[late-interaction]]
- [[knowledge-distillation]]

## Sources

- [[yang-2024-distillation-2405-00977]]
