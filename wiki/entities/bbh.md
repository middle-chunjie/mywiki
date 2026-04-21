---
type: entity
title: BBH
slug: bbh
date: 2026-04-20
entity_type: benchmark
aliases: [Big-Bench Hard, BIG-Bench Hard]
tags: []
---

## Description

BBH is the challenging reasoning and in-context learning benchmark used in [[jiang-2023-llmlingua-2310-05736]] to test whether compressed prompts preserve difficult multi-step reasoning behavior.

## Key Contributions

- Provides a hard evaluation setting where LLMLingua reaches `70.11` EM at `3x` compression and `56.85` EM at `7x` compression.
- Helps show that prompt compression can retain more reasoning signal than Selective-Context and GPT-4 generation baselines.

## Related Concepts

- [[in-context-learning]]
- [[chain-of-thought]]
- [[prompt-compression]]

## Sources

- [[jiang-2023-llmlingua-2310-05736]]
