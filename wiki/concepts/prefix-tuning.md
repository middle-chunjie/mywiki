---
type: concept
title: Prefix-Tuning
slug: prefix-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [prefix tuning, 前缀微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prefix-Tuning** (前缀微调) — a parameter-efficient fine-tuning method that prepends learned continuous key-value states or soft prompt vectors to a model's attention computation instead of updating all backbone weights.

## Key Points

- The paper treats Prefix-Tuning as the prompt-based PEFT baseline within the LLM-Adapters framework.
- Its core formulation augments attention with learned prefix states `P_K` and `P_V`, rather than inserting a bottleneck MLP inside the backbone.
- In the configuration study on `LLaMA-7B`, the best setting is `vt = 10`, which yields `42.0%` average accuracy on six math reasoning datasets.
- Prefix-Tuning is consistently weaker than Series Adapter, Parallel Adapter, and LoRA in the reported reasoning experiments, especially on stronger LLaMA backbones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hu-2023-llmadapters-2304-01933]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hu-2023-llmadapters-2304-01933]].
