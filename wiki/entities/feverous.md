---
type: entity
title: FEVEROUS
slug: feverous
date: 2026-04-20
entity_type: dataset
aliases: [Feverous]
tags: []
---

## Description

FEVEROUS is a fact-verification benchmark over both unstructured and structured evidence. In [[shao-2023-enhancing-2305-15294]], it is the paper's evaluation dataset for retrieval-augmented verification under few-shot prompting.

## Key Contributions

- Provides the fact-verification setting in which Iter-RetGen is compared against Direct prompting, ReAct, Self-Ask, and DSP.
- Shows that Iter-RetGen is competitive with Self-Ask at `68.8 Acc†` with `T = 2`, and reaches `69.5 Acc†` after generation-augmented retriever adaptation.
- Serves as one of the two datasets used to train and validate the reranker-to-retriever distillation procedure.

## Related Concepts

- [[fact-verification]]
- [[dense-retrieval]]
- [[retrieval-augmented-generation]]

## Sources

- [[shao-2023-enhancing-2305-15294]]
