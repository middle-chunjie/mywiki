---
type: entity
title: ConvAug
slug: convaug
date: 2026-04-20
entity_type: method
aliases: []
tags: []
---

## Description

ConvAug is the framework introduced by this paper for generalizing conversational dense retrieval through LLM-based multi-level data augmentation and contrastive training. It couples controlled synthetic conversation generation with difficulty-aware sample selection.

## Key Contributions

- Generates positive and hard-negative conversational variants across token, turn, and conversation levels.
- Introduces cognition-aware prompting and a difficulty-adaptive filter for higher-quality augmentation.
- Improves ANCE, Conv-SPLADE, and LeCoRE on normal and zero-shot conversational retrieval benchmarks.

## Related Concepts

- [[conversational-dense-retrieval]]
- [[data-augmentation]]
- [[contrastive-learning]]

## Sources

- [[chen-2024-generalizing-2402-07092]]
