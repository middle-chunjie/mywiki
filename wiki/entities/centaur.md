---
type: entity
title: CENTaUR
slug: centaur
date: 2026-04-20
entity_type: model
aliases: [CENTaUR]
tags: []
---

## Description

CENTaUR is the paper's adapted cognitive model: a linear choice-prediction layer trained on top of frozen LLaMA embeddings extracted from text descriptions of psychological experiments.

## Key Contributions

- Outperforms raw LLaMA and domain-specific cognitive baselines on both choices13k and the horizon task.
- Supports participant-level random effects, improving fit for individual human behavior.
- Transfers to an unseen experiential-symbolic task after multi-task training.

## Related Concepts

- [[cognitive-model]]
- [[linear-probing]]
- [[cross-task-generalization]]

## Sources

- [[unknown-nd-turning-2306-03917]]
