---
type: entity
title: mC4
slug: mc4
date: 2026-04-20
entity_type: dataset
aliases: [mC4, multilingual C4]
tags: []
---

## Description

mC4 is the multilingual extension of the C4 web corpus and is the training dataset used in [[xiao-2026-embedding-2602-11047]] for learning the conditional diffusion inversion decoder. The paper samples `2M` multilingual examples, each truncated to `32` tokens.

## Key Contributions

- Provides the multilingual training distribution for all three inversion models in the paper.
- Supports the paper's evaluation claim that the method generalizes across ten languages.
- Supplies paired text needed to cache target embeddings from each encoder during training.

## Related Concepts

- [[text-embedding]]
- [[embedding-inversion]]
- [[privacy-leakage]]

## Sources

- [[xiao-2026-embedding-2602-11047]]
