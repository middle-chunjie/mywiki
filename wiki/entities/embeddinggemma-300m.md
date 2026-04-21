---
type: entity
title: EmbeddingGemma-300m
slug: embeddinggemma-300m
date: 2026-04-20
entity_type: tool
aliases: [EmbeddingGemma-300m, EmbeddingGemma]
tags: []
---

## Description

EmbeddingGemma-300m is a `300M`-parameter text embedding model used as one of the evaluation encoders in [[xiao-2026-embedding-2602-11047]]. It produces `768`-dimensional embeddings and gives a third architectural test bed for the paper's inversion method.

## Key Contributions

- Supplies one of the three target embedding spaces used in the paper's experiments.
- Achieves `78.8%` token accuracy under sequential greedy decoding.
- Shows that the attack remains effective beyond the two `1024`-dimensional encoder settings.

## Related Concepts

- [[text-embedding]]
- [[embedding-inversion]]
- [[black-box-attack]]

## Sources

- [[xiao-2026-embedding-2602-11047]]
