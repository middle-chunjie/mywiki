---
type: entity
title: Atlas
slug: atlas
date: 2026-04-20
entity_type: model
aliases: []
tags: []
---

## Description

Atlas is a retrieval-augmented language model whose reader uses a T5-based Fusion-in-Decoder architecture. [[unknown-nd-btrbinary-2310-01329]] uses Atlas base and large as the main host models for evaluating BTR.

## Key Contributions

- Provides the baseline reader architecture on top of which BTR is implemented and compared.
- Supplies both base (`220M`) and large (`770M`) model variants for the paper's speed and accuracy trade-off study.
- Shows that BTR can preserve most Atlas performance while materially improving reader throughput.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[fusion-in-decoder]]
- [[passage-representation-caching]]

## Sources

- [[unknown-nd-btrbinary-2310-01329]]
