---
type: entity
title: StarEncoder
slug: starencoder
date: 2026-04-20
entity_type: tool
aliases: [StarEncoder, star-encoder]
tags: [model, baseline, code-embedding]
---

## Description

StarEncoder is a bidirectional encoder pretrained on 86 programming languages from The Stack using MLM and next-sentence prediction objectives. It serves as a strong baseline for code representation tasks in [[zhang-2024-code-2402-01935]].

## Key Contributions

- Scales code encoder pretraining to 86 programming languages from The Stack corpus.
- Uses MLM and next-sentence prediction as pretraining objectives, following BERT-style training.
- Serves as a key same-category comparison point against CodeSage (both are large-scale, encoder-only code models).

## Related Concepts

- [[code-representation-learning]]
- [[masked-language-modeling]]
- [[code-pretrained-model]]

## Sources

- [[zhang-2024-code-2402-01935]]
