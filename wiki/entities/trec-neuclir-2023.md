---
type: entity
title: TREC NeuCLIR 2023
slug: trec-neuclir-2023
date: 2026-04-20
entity_type: dataset
aliases: [NeuCLIR 2023, TREC 2023 NeuCLIR]
tags: []
---

## Description

TREC NeuCLIR 2023 is the bilingual topic development benchmark used in [[yang-2024-distillation-2405-00977]] for multilingual retrieval over Chinese, Persian, and Russian documents. The paper uses it to test whether MTD remains effective when topics are designed to be less culturally tied to a single language.

## Key Contributions

- Demonstrates that MTD continues to outperform `ColBERT-X MTT`, reaching up to `0.404 nDCG@20`, `0.372 MAP`, and `0.877 Recall@1000`.
- Supports the claim that the three language-mixing strategies are close in effectiveness on multilingual ranking metrics.

## Related Concepts

- [[multilingual-retrieval]]
- [[multilingual-dense-retrieval]]
- [[language-mixing-strategy]]

## Sources

- [[yang-2024-distillation-2405-00977]]
