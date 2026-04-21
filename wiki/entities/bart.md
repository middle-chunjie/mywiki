---
type: entity
title: BART
slug: bart
date: 2026-04-20
entity_type: tool
aliases: [BART-large]
tags: []
---

## Description

BART is the denoising sequence-to-sequence Transformer backbone used in [[wang-2024-improving-2401-04361]] for both the paraphrase model and the KGD encoder-decoder initialization.

## Key Contributions

- Supplies the `12`-layer encoder and `12`-layer decoder initialization for the KGD model.
- Serves as the base architecture for the entity-guided paraphrase generator.
- Provides the strongest perplexity baseline, highlighting the trade-off between robustness gains and likelihood.

## Related Concepts

- [[knowledge-grounded-dialogue]]
- [[cross-attention]]
- [[data-augmentation]]

## Sources

- [[wang-2024-improving-2401-04361]]
