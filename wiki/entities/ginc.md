---
type: entity
title: GINC
slug: ginc
date: 2026-04-20
entity_type: dataset
aliases: [Generative IN-Context learning dataset]
tags: []
---

## Description

GINC is the synthetic dataset introduced in [[xie-2022-explanation-2111-02080]] for studying in-context learning under a controlled mixture-of-HMM pretraining distribution. It is designed so the latent concept structure and prompt mismatch are fully manipulable.

## Key Contributions

- Provides a small-scale testbed where both Transformers and LSTMs exhibit in-context learning.
- Enables ablations on concept mixture structure, unseen concepts, model scale, and prompt ordering.
- Makes it possible to separate pretraining loss from downstream in-context accuracy in a controlled setting.

## Related Concepts

- [[in-context-learning]]
- [[hidden-markov-model]]
- [[distribution-shift]]

## Sources

- [[xie-2022-explanation-2111-02080]]
