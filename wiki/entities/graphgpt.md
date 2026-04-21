---
type: entity
title: GraphGPT
slug: graphgpt
date: 2026-04-20
entity_type: tool
aliases: [GraphGPT framework]
tags: []
---

## Description

GraphGPT is the graph-oriented large language model framework introduced in [[tang-2024-graphgpt-2310-13023]]. It aligns graph encoder outputs with an LLM through text-graph grounding, projector tuning, and graph-specific instructions.

## Key Contributions

- Introduces a dual-stage graph instruction tuning pipeline for graph matching and downstream graph tasks.
- Uses a lightweight projector to connect frozen graph encoders and Vicuna-based LLMs efficiently.
- Combines graph grounding with CoT distillation to improve zero-shot transfer on citation graphs.

## Related Concepts

- [[graph-instruction-tuning]]
- [[text-graph-grounding]]
- [[chain-of-thought-distillation]]

## Sources

- [[tang-2024-graphgpt-2310-13023]]
