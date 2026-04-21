---
type: entity
title: zsRE Dataset
slug: zsre-dataset
date: 2026-04-20
entity_type: dataset
aliases: [zsRE, Zero-Shot Relation Extraction, zsRE slot-filling]
tags: [dataset, slot-filling, zero-shot, relation-extraction]
---

## Description

zsRE (Zero-Shot Relation Extraction) is a dataset introduced by Levy et al. (2017) that frames relation extraction as reading comprehension, enabling zero-shot evaluation across unseen relation types by converting relation triples into natural-language questions. It is included in the KILT benchmark as a slot-filling task.

## Key Contributions

- Frames relation extraction as machine reading comprehension, using question-answer pairs generated from relation templates to enable zero-shot generalization.
- Included in the [[kilt-benchmark]] as a slot-filling dataset evaluated with KILT-AC (Accuracy).
- In [[zamani-2024-stochastic]], Stochastic RAG shows the largest absolute gain on zsRE: FiD-Light XL improves from KILT-AC 84.0 to 87.0, also outperforming GripRank by a large margin.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[kilt-score]]

## Sources

- [[zamani-2024-stochastic]]
