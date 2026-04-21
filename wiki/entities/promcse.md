---
type: entity
title: PromCSE
slug: promcse
date: 2026-04-20
entity_type: tool
aliases: [Prompt-based Contrastive Learning for Sentence Embeddings]
tags: []
---

## Description

PromCSE is the sentence embedding method introduced in [[jiang-2022-improved-2203-06875]]. It freezes a pretrained encoder, learns multi-layer soft prompts, and optionally adds an energy-based hinge loss in supervised training.

## Key Contributions

- Replaces full-model sentence contrastive fine-tuning with layer-wise soft prompt tuning.
- Improves unsupervised average STS from `76.25` to `78.49` over SimCSE on BERTbase.
- Reaches `71.2` on CxC-STS in the unsupervised setting and `74.0` with supervised EH training, showing stronger robustness under domain shift.

## Related Concepts

- [[sentence-embedding]]
- [[contrastive-learning]]
- [[soft-prompt-tuning]]
- [[energy-based-learning]]

## Sources

- [[jiang-2022-improved-2203-06875]]
