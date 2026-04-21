---
type: entity
title: InfoSeekQA
slug: infoseekqa
date: 2026-04-20
entity_type: tool
aliases: [InfoSeek QA, InfoSeekQA dataset]
tags: []
---

## Description

InfoSeekQA is a large-scale deep research benchmark comprising over 50K question–answer pairs requiring hierarchical multi-step reasoning and iterative information acquisition. In [[zhou-2026-retrieve-2604-04949]], the top 10K queries with verified ground-truth answers are used as seed data for trajectory generation, with answer correctness verified by Qwen3-30B-A3B-Thinking-2507.

## Key Contributions

- Provides the seed query set for generating 26,482 agent trajectories used to train LRAT retrievers.
- Designed to produce significantly longer interaction trajectories than traditional QA datasets, making it well-suited for deep research agent trajectory collection.
- Uses the Wiki-25-Dump corpus (11.2M document chunks of 512 tokens) as the retrieval target.

## Related Concepts

- [[deep-research-agent]]
- [[agentic-search]]
- [[information-retrieval]]

## Sources

- [[zhou-2026-retrieve-2604-04949]]
