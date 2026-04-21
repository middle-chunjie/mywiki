---
type: entity
title: Tongyi-DeepResearch-30B
slug: tongyi-deepresearch-30b
date: 2026-04-20
entity_type: tool
aliases: [Tongyi-DeepResearch-30B-A3B, Tongyi DeepResearch 30B]
tags: []
---

## Description

Tongyi-DeepResearch-30B-A3B is an open-source deep research agent developed by [[tongyi-lab]], used in [[zhou-2026-retrieve-2604-04949]] as the trajectory-generation agent for LRAT training data collection. It supports over 100 interaction steps and is designed for long-horizon, multi-step information-seeking tasks.

## Key Contributions

- Generates the 26,482 agent trajectories (10K InfoSeekQA queries × 4 retrievers) used to train LRAT retrievers.
- Acts as the evaluation agent for in-domain InfoSeek-Eval benchmarks in LRAT experiments.
- Demonstrates that LRAT-trained retrievers improve its own task success rate (52.7 → 68.0 with Qwen3-Emb).

## Related Concepts

- [[deep-research-agent]]
- [[agentic-search]]
- [[agent-trajectory]]

## Sources

- [[zhou-2026-retrieve-2604-04949]]
