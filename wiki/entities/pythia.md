---
type: entity
title: Pythia
slug: pythia
date: 2026-04-20
entity_type: model
aliases: [Pythia-2.8B, Pythia-6.9B, Pythia-12B, Pythia-160M]
tags: []
---

## Description

Pythia is both an evaluated LLM family in the paper and the codebase used for the authors' `160M`-parameter sink-token pretraining study. It provides the controlled setting for testing whether sink-token pretraining preserves standard LM quality while improving streaming behavior.

## Key Contributions

- Supplies the `160M` training setup used to compare vanilla attention, zero sink, and learnable sink-token variants.
- Shows long-stream perplexity recovery from `21.62` to `12.09` on Pythia-12B when four sink tokens are preserved.

## Related Concepts

- [[sink-token]]
- [[softmax-off-by-one]]
- [[attention-sink]]

## Sources

- [[xiao-2024-efficient-2309-17453]]
