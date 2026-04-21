---
type: entity
title: T2Ranking
slug: t2ranking
date: 2026-04-20
entity_type: dataset
aliases: [T2Ranking Benchmark]
tags: []
---

## Description

T2Ranking is the Chinese passage-ranking benchmark used in [[fang-2024-scaling-2403-18684]]. The paper uses it to test whether dense-retrieval scaling laws also hold beyond the English MS MARCO setting.

## Key Contributions

- Provides `300k+` queries and `2M+` unique passages for the paper's Chinese dense-retrieval experiments.
- Supports the ERNIE-based evaluation of model-size and data-size scaling laws.
- Confirms strong power-law fits alongside MS MARCO, with `R^2` up to `0.999` for model-size scaling.

## Related Concepts

- [[dense-retrieval]]
- [[contrastive-perplexity]]
- [[power-law-scaling]]

## Sources

- [[fang-2024-scaling-2403-18684]]
