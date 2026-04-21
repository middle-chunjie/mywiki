---
type: entity
title: MetaMath
slug: metamath
date: 2026-04-20
entity_type: dataset
aliases: [MetaMath, MetaMath dataset]
tags: []
---

## Description

MetaMath is the math instruction-tuning dataset used in [[wang-2024-mathshepherd-2312-08935]] to train generators and completers before reward-model construction. It supplies the base supervised reasoning capability for the open-source LLMs studied in the paper.

## Key Contributions

- Provides the supervised fine-tuning data for the generator and completer models.
- Serves as the base training corpus before verification and [[step-by-step-ppo]] experiments.

## Related Concepts

- [[mathematical-reasoning]]
- [[process-supervision]]
- [[large-language-model]]

## Sources

- [[wang-2024-mathshepherd-2312-08935]]
