---
type: concept
title: Chain-of-Thought Distillation
slug: chain-of-thought-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [CoT distillation, 思维链蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Chain-of-Thought Distillation** (思维链蒸馏) — a training strategy that transfers step-by-step reasoning traces from a stronger teacher model into a smaller or more specialized student model.

## Key Points

- GraphGPT uses GPT-3.5 to generate reasoning traces for node-specific graph tasks and mixes those traces into instruction data.
- The paper motivates CoT distillation as a way to improve robustness under distribution shift and varying class spaces across graph datasets.
- Distillation is added after the graph-token interface has already been established, so it augments reasoning rather than replacing structural alignment.
- The reported `-cot` variants outperform standard-instruction variants on harder transfers such as Arxiv -> Cora.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-graphgpt-2310-13023]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-graphgpt-2310-13023]].
