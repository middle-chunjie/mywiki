---
type: entity
title: PaLM 2-S*
slug: palm-2-s
date: 2026-04-20
entity_type: model
aliases: [PaLM 2-S, "PaLM 2-S*", Palm 2 S star]
tags: []
---

## Description

PaLM 2-S* is the base language model used throughout [[snell-2024-scaling-2408-03314]] to study how extra inference budget should be allocated on reasoning tasks.

## Key Contributions

- Serves as the proposal model for both best-of-`N` sampling and sequential revision experiments.
- Provides the backbone for the trained [[process-reward-model]] and revision-specific verifier.
- Defines the model-relative notion of [[question-difficulty]] used by the paper.

## Related Concepts

- [[test-time-compute]]
- [[process-reward-model]]
- [[question-difficulty]]

## Sources

- [[snell-2024-scaling-2408-03314]]
