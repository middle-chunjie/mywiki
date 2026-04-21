---
type: entity
title: M2-BERT
slug: m2-bert
date: 2026-04-20
entity_type: tool
aliases: [M2 BERT, M2-BERT-32k, M2-BERT retrieval encoder]
tags: []
---

## Description

M2-BERT is the `80M`-parameter long-context retrieval encoder introduced in [[saad-falcon-2024-benchmarking-2402-07440]]. It is built on Monarch Mixer and scales to documents up to `32k` tokens.

## Key Contributions

- Achieves the paper's best LoCoV1 result, outperforming strong Transformer retrievers and BM25.
- Demonstrates that a state-space encoder can be both more accurate on long-context retrieval and substantially faster at embedding generation.
- Serves as the paper's vehicle for studying mixed-length pretraining and OPL-based fine-tuning.

## Related Concepts

- [[state-space-model]]
- [[dense-retrieval]]
- [[orthogonal-projection-loss]]

## Sources

- [[saad-falcon-2024-benchmarking-2402-07440]]
