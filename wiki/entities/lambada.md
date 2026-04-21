---
type: entity
title: LAMBADA
slug: lambada
date: 2026-04-20
entity_type: dataset
aliases: [LAMBADA dataset]
tags: []
---

## Description

LAMBADA is a broad-context word prediction benchmark referenced in [[xie-2022-explanation-2111-02080]] both as prior evidence for GPT-3's in-context learning and as the basis of the appendix experiment on example length. In this paper it serves as a real-world sanity check for the theory's claim that longer examples can help by revealing more latent context.

## Key Contributions

- Anchors the paper's discussion of large-scale in-context learning behavior in a real benchmark rather than only the synthetic GINC setup.
- Supports the appendix result that `5` longer examples outperform `5` shorter examples (`70.7%` vs. `69.8%`) on a filtered test split.
- Helps distinguish gains from richer contextual evidence versus gains from simply increasing total prompt length.

## Related Concepts

- [[few-shot-learning]]
- [[in-context-learning]]
- [[large-language-model]]

## Sources

- [[xie-2022-explanation-2111-02080]]
