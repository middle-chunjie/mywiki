---
type: entity
title: DeepSeek R1
slug: deepseek-r1
date: 2026-04-20
entity_type: model
aliases: [R1, Deepseek R1, DeepSeek-R1]
tags: []
---

## Description

DeepSeek R1 is the reasoning language model used as the teacher model in [[weller-2025-rank-2502-18418]]. The paper queries it on MS MARCO pairs to generate explicit reasoning traces and relevance labels for distillation.

## Key Contributions

- Supplies the `635,264` teacher-generated reasoning traces used to train Rank1.
- Provides both rationales and final relevance labels, enabling supervised distillation rather than RL-heavy training.

## Related Concepts

- [[large-language-model]]
- [[knowledge-distillation]]
- [[reasoning-trace]]

## Sources

- [[weller-2025-rank-2502-18418]]
