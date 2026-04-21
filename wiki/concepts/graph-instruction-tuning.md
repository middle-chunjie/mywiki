---
type: concept
title: Graph Instruction Tuning
slug: graph-instruction-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [graph instruction tuning, 图指令微调]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Graph Instruction Tuning** (图指令微调) — an instruction-tuning strategy that adapts a language model to graph-structured tasks by injecting graph tokens and optimizing on graph-specific instructions instead of text-only prompts.

## Key Points

- GraphGPT uses a dual-stage graph instruction tuning pipeline rather than directly fine-tuning the full LLM on graph tasks.
- The first stage is self-supervised and teaches the model to associate graph tokens with textual node identities through graph matching.
- The second stage specializes the learned graph-token interface for downstream tasks such as node classification and link prediction.
- The paper freezes the LLM and graph encoder, tuning only the projector, which makes the instruction-tuning procedure much cheaper than end-to-end updating.
- The authors attribute much of the framework's zero-shot transfer performance to this graph-native instruction format.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-graphgpt-2310-13023]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-graphgpt-2310-13023]].
