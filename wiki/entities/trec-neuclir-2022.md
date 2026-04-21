---
type: entity
title: TREC NeuCLIR 2022
slug: trec-neuclir-2022
date: 2026-04-20
entity_type: dataset
aliases: [NeuCLIR 2022, TREC 2022 NeuCLIR]
tags: []
---

## Description

TREC NeuCLIR 2022 is the multilingual retrieval benchmark used in [[yang-2024-distillation-2405-00977]] with English queries over Chinese, Persian, and Russian documents. It is the paper's hardest evaluation setting and highlights the gains of multilingual training over zero-shot transfer.

## Key Contributions

- Exposes a larger language gap than CLEF, making it a strong test of multilingual ranking calibration.
- Shows MTD improving over `ColBERT-X MTT` from `0.375` to `0.474` in `nDCG@20` and from `0.236` to `0.347` in `MAP`.

## Related Concepts

- [[multilingual-retrieval]]
- [[multilingual-dense-retrieval]]
- [[zero-shot-retrieval]]

## Sources

- [[yang-2024-distillation-2405-00977]]
