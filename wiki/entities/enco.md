---
type: entity
title: EnCo
slug: enco
date: 2026-04-20
entity_type: tool
aliases: [entity-based contrastive learning]
tags: []
---

## Description

EnCo is the entity-based contrastive learning framework proposed in [[wang-2024-improving-2401-04361]] for improving robustness in [[knowledge-grounded-dialogue]] under noisy contexts and imperfect knowledge graphs.

## Key Contributions

- Constructs positive samples with entity-guided paraphrasing and truncated knowledge.
- Constructs hard negative samples by entity deletion or replacement in both context and KG.
- Combines cross-entropy supervision with a contrastive objective to improve robustness and few-shot performance.

## Related Concepts

- [[knowledge-grounded-dialogue]]
- [[contrastive-learning]]
- [[data-augmentation]]
- [[named-entity-recognition]]

## Sources

- [[wang-2024-improving-2401-04361]]
