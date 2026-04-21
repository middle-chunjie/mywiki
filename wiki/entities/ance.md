---
type: entity
title: ANCE
slug: ance
date: 2026-04-20
entity_type: model
aliases: [Approximate Nearest Neighbor Negative Contrastive Learning]
tags: []
---

## Description

ANCE is the base dense retriever used to instantiate ConvAug in this paper. The authors build their conversational context encoder on top of ANCE and then improve it through augmentation and multi-task contrastive training.

## Key Contributions

- Serves as the backbone retriever for the main ConvAug experiments.
- Provides the baseline architecture against which ConvAug demonstrates gains on QReCC, TopiOCQA, and CAsT.

## Related Concepts

- [[conversational-dense-retrieval]]
- [[contrastive-learning]]
- [[zero-shot-generalization]]

## Sources

- [[chen-2024-generalizing-2402-07092]]
